"""Macro regime overlay: USD/INR and US 10Y treasury yield, from Alpha Vantage.

The stock-level trend filter (strategy/factors.py) is bottom-up -- it reacts
once individual names cross their own 200-day SMA. This adds a top-down
signal: a fast-weakening rupee or a sharp spike in US yields tends to precede
FII outflows from Indian equities as a whole, ahead of any single stock's
chart turning down. When that shows up, we raise the cash buffer instead of
staying fully invested.

Both series update at most daily/monthly, so results are cached on disk and
only re-fetched every CACHE_MAX_AGE_DAYS -- keeps this well inside Alpha
Vantage's free-tier rate limit (5 calls/min, 25/day).
"""
import os
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://www.alphavantage.co/query"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data_cache", "macro")
CACHE_MAX_AGE_DAYS = 25

INR_LOOKBACK_DAYS = 90  # ~3 months
INR_WEAKENING_THRESHOLD = 0.03  # USD/INR up >3% over the lookback -> risk-off

YIELD_LOOKBACK_MONTHS = 3
YIELD_SPIKE_THRESHOLD_BPS = 75  # US 10Y up >75bps over the lookback -> risk-off

RISK_OFF_CASH_BUFFER = 0.25


def _api_key() -> str:
    return os.environ["ALPHA_VANTAGE_API_KEY"]


def _cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, f"{name}.csv")


def _is_stale(path: str) -> bool:
    if not os.path.exists(path):
        return True
    age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
    return age > timedelta(days=CACHE_MAX_AGE_DAYS)


def _get(params: dict) -> dict:
    resp = requests.get(BASE_URL, params={**params, "apikey": _api_key()}, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    if "Note" in data or "Information" in data or "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data}")
    return data


def fetch_usdinr(refresh: bool = False) -> pd.Series:
    """Daily USD/INR close (Alpha Vantage FX_DAILY)."""
    path = _cache_path("usdinr")
    if not refresh and not _is_stale(path):
        return pd.read_csv(path, index_col="date", parse_dates=True)["close"]

    data = _get({"function": "FX_DAILY", "from_symbol": "USD", "to_symbol": "INR", "outputsize": "full"})
    df = pd.DataFrame(data["Time Series FX (Daily)"]).T
    df.index = pd.to_datetime(df.index)
    close = df["4. close"].astype(float).sort_index()
    close.name = "close"

    os.makedirs(CACHE_DIR, exist_ok=True)
    close.to_frame().to_csv(path, index_label="date")
    return close


def fetch_treasury_yield_10y(refresh: bool = False) -> pd.Series:
    """Monthly US 10-year treasury yield (Alpha Vantage TREASURY_YIELD)."""
    path = _cache_path("treasury_10y")
    if not refresh and not _is_stale(path):
        return pd.read_csv(path, index_col="date", parse_dates=True)["value"]

    data = _get({"function": "TREASURY_YIELD", "interval": "monthly", "maturity": "10year"})
    df = pd.DataFrame(data["data"])
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = df.dropna().set_index("date")["value"].sort_index()

    os.makedirs(CACHE_DIR, exist_ok=True)
    series.to_frame().to_csv(path, index_label="date")
    return series


def regime_signal(as_of=None) -> dict:
    """Risk-on/off read as of a given date (defaults to the latest cached data).

    Risk-off if either:
      - USD/INR has risen (rupee weakened) more than INR_WEAKENING_THRESHOLD
        over the trailing INR_LOOKBACK_DAYS, or
      - the US 10Y yield has risen more than YIELD_SPIKE_THRESHOLD_BPS over
        the trailing YIELD_LOOKBACK_MONTHS.
    """
    usdinr = fetch_usdinr()
    yield_10y = fetch_treasury_yield_10y()
    if as_of is not None:
        usdinr = usdinr.loc[:as_of]
        yield_10y = yield_10y.loc[:as_of]

    inr_change_pct = None
    inr_risk_off = False
    if len(usdinr) > 1:
        past = usdinr.loc[: usdinr.index[-1] - timedelta(days=INR_LOOKBACK_DAYS)]
        if len(past):
            inr_change_pct = usdinr.iloc[-1] / past.iloc[-1] - 1.0
            inr_risk_off = inr_change_pct > INR_WEAKENING_THRESHOLD

    yield_change_bps = None
    yield_risk_off = False
    if len(yield_10y) > 1:
        past = yield_10y.loc[: yield_10y.index[-1] - pd.DateOffset(months=YIELD_LOOKBACK_MONTHS)]
        if len(past):
            yield_change_bps = (yield_10y.iloc[-1] - past.iloc[-1]) * 100
            yield_risk_off = yield_change_bps > YIELD_SPIKE_THRESHOLD_BPS

    return {
        "as_of": usdinr.index[-1] if len(usdinr) else None,
        "usdinr": usdinr.iloc[-1] if len(usdinr) else None,
        "usdinr_change_pct": inr_change_pct,
        "yield_10y": yield_10y.iloc[-1] if len(yield_10y) else None,
        "yield_10y_change_bps": yield_change_bps,
        "risk_off": inr_risk_off or yield_risk_off,
    }


def cash_buffer(as_of=None, normal: float = 0.05) -> float:
    """Cash buffer to hold: elevated when the macro regime reads risk-off."""
    return RISK_OFF_CASH_BUFFER if regime_signal(as_of)["risk_off"] else normal


if __name__ == "__main__":
    signal = regime_signal()
    print(signal)
    print("Cash buffer:", cash_buffer())
