"""Fetch and cache daily OHLCV history from Groww for backtesting/signals."""
import os
import time
from datetime import datetime, timedelta

import pandas as pd
from growwapi import GrowwAPI

from client import get_client
from strategy.universe import groww_symbol

CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache", "candles")
MAX_DAYS_PER_REQUEST = 175  # Groww caps 1day interval at 180 days per request
COLUMNS = ["open", "high", "low", "close", "volume"]


def _cache_path(trading_symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{trading_symbol}.csv")


def _fetch_chunk(groww, gsymbol: str, start: datetime, end: datetime, retries: int = 6) -> pd.DataFrame:
    delay = 2.0
    for attempt in range(retries):
        try:
            resp = groww.get_historical_candles(
                exchange=GrowwAPI.EXCHANGE_NSE,
                segment=GrowwAPI.SEGMENT_CASH,
                groww_symbol=gsymbol,
                start_time=start.strftime("%Y-%m-%d %H:%M:%S"),
                end_time=end.strftime("%Y-%m-%d %H:%M:%S"),
                candle_interval="1day",
            )
            break
        except Exception as e:
            if "Rate limit" in str(e) and attempt < retries - 1:
                time.sleep(delay)
                delay = min(delay * 1.8, 30)
                continue
            raise
    candles = resp.get("candles", [])
    if not candles:
        return pd.DataFrame(columns=COLUMNS)
    df = pd.DataFrame(candles, columns=["date", "open", "high", "low", "close", "volume", "oi"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date")[COLUMNS]
    return df


def fetch_history(
    trading_symbol: str,
    start: datetime,
    end: datetime,
    groww=None,
    use_cache: bool = True,
    sleep_between_calls: float = 0.6,
) -> pd.DataFrame:
    """Daily OHLCV for one symbol between start and end (inclusive), cached on disk.

    Incrementally extends the cache: if it's stale at the end, only the
    missing tail is fetched from the API. A cache whose earliest row is after
    `start` is trusted as "this symbol simply has no data before that" (e.g.
    a stock listed after `start`) rather than triggering a full re-fetch --
    every cache file on disk was built from a request starting at `start`.
    """
    path = _cache_path(trading_symbol)
    cached = pd.DataFrame(columns=COLUMNS)
    fetch_from = start

    if use_cache and os.path.exists(path):
        cached = pd.read_csv(path, index_col="date", parse_dates=True)
        if not cached.empty:
            if cached.index.max() >= end - timedelta(days=3):
                return cached.loc[start:end]
            fetch_from = cached.index.max() + timedelta(days=1)

    groww = groww or get_client()
    gsymbol = groww_symbol(trading_symbol)

    chunks = []
    cursor = fetch_from
    while cursor <= end:
        chunk_end = min(cursor + timedelta(days=MAX_DAYS_PER_REQUEST), end)
        df = _fetch_chunk(groww, gsymbol, cursor, chunk_end)
        if not df.empty:
            chunks.append(df)
        cursor = chunk_end + timedelta(days=1)
        time.sleep(sleep_between_calls)

    new_data = pd.concat(chunks).sort_index() if chunks else pd.DataFrame(columns=COLUMNS)
    full = pd.concat([cached, new_data]).sort_index() if not cached.empty else new_data
    full = full[~full.index.duplicated(keep="last")]

    if use_cache and not full.empty:
        os.makedirs(CACHE_DIR, exist_ok=True)
        full.to_csv(path, index_label="date")

    return full.loc[start:end]


def open_price_quality(history: dict[str, pd.DataFrame]) -> dict:
    """Measure how trustworthy the `open` column is, by year.

    Groww's cached daily candles return open == previous close for most rows
    from 2025 onward (98.9% in this dataset) and omit open entirely on some
    days. A synthetic open makes "execute at the next open" collapse into
    "execute at the signal bar's close" -- reintroducing the exact same-bar
    lookahead that next-bar execution is meant to remove. Callers that want to
    fill at the open must check this first.

    Returned `by_year` maps year -> {"n", "synthetic_share"}. Judge per year,
    not on the overall average: the corruption here is concentrated in 2025+,
    and a single bad year invalidates a backtest that spans it.
    """
    frames = []
    for df in history.values():
        df = df.sort_index()
        frames.append(pd.DataFrame({"open": df["open"], "prev_close": df["close"].shift(1)}))
    empty = {"rows": 0, "synthetic_share": float("nan"), "by_year": {}}
    if not frames:
        return empty

    combined = pd.concat(frames)
    combined = combined[combined["open"].notna() & combined["prev_close"].notna()]
    if combined.empty:
        return empty

    synthetic = combined["open"] == combined["prev_close"]
    grouped = synthetic.groupby(combined.index.year)
    return {
        "rows": len(combined),
        "synthetic_share": float(synthetic.mean()),
        "by_year": {
            int(year): {"n": int(g.size), "synthetic_share": float(g.mean())}
            for year, g in grouped
        },
    }


def suspect_open_years(
    quality: dict, max_synthetic_share: float = 0.25, min_rows: int = 200
) -> list[tuple[int, int, float]]:
    """Years whose `open` column looks synthetic, as (year, n, synthetic_share)."""
    return [
        (year, stats["n"], stats["synthetic_share"])
        for year, stats in sorted(quality.get("by_year", {}).items())
        if stats["n"] >= min_rows and stats["synthetic_share"] > max_synthetic_share
    ]


class _LazyClient:
    """Authenticates on first actual use, so a fully-cached run needs no credentials.

    fetch_history() only touches the client when it has to hit the API, so
    passing this through lets an offline backtest run entirely off data_cache/
    without valid Groww API keys.
    """

    def __init__(self):
        self._client = None

    def __getattr__(self, name):
        if self._client is None:
            self._client = get_client()
        return getattr(self._client, name)


def fetch_universe_history(
    symbols: list[str], start: datetime, end: datetime, use_cache: bool = True
) -> dict[str, pd.DataFrame]:
    """Fetch history for many symbols, reusing one authenticated client."""
    groww = _LazyClient()
    out = {}
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_history(sym, start, end, groww=groww, use_cache=use_cache)
            if not df.empty:
                out[sym] = df
        except Exception as e:
            print(f"  [{i}/{len(symbols)}] {sym}: skipped ({e})")
            continue
        if i % 25 == 0:
            print(f"  fetched {i}/{len(symbols)} symbols...")
    return out


if __name__ == "__main__":
    from strategy.universe import get_universe

    end = datetime(2026, 8, 28)
    start = datetime(2020, 1, 1)
    symbols = get_universe()[:5]
    data = fetch_universe_history(symbols, start, end)
    for sym, df in data.items():
        print(sym, df.shape, df.index.min().date(), df.index.max().date())
