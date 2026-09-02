"""Investable universe: Nifty 200 constituents, pulled live from NSE."""
import io
import os

import pandas as pd
import requests

NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"
CACHE_PATH = os.path.join(os.path.dirname(__file__), "data_cache", "nifty200.csv")


def get_universe(refresh: bool = False) -> list[str]:
    """Return NSE trading symbols for the current Nifty 200 constituents.

    Fetches the live list from NSE archives and caches it locally, since that
    endpoint is occasionally flaky. Falls back to the cache if the fetch fails.
    """
    if not refresh and os.path.exists(CACHE_PATH):
        return _read_symbols(CACHE_PATH)

    try:
        resp = requests.get(
            NIFTY200_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15
        )
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        return df["Symbol"].tolist()
    except Exception:
        if os.path.exists(CACHE_PATH):
            return _read_symbols(CACHE_PATH)
        raise


def _read_symbols(path: str) -> list[str]:
    return pd.read_csv(path)["Symbol"].tolist()


def groww_symbol(trading_symbol: str, exchange: str = "NSE") -> str:
    return f"{exchange}-{trading_symbol}"


if __name__ == "__main__":
    symbols = get_universe(refresh=True)
    print(f"{len(symbols)} symbols in universe")
    print(symbols[:10])
