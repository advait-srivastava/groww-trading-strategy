"""Investable universe: Nifty 200 constituents, with point-in-time support.

Survivorship bias
-----------------
NSE publishes only the *current* Nifty 200 constituent list. There is no public
dated archive -- `ind_nifty200list.csv` resolves, but dated variants
(`ind_nifty200list_01012022.csv`, `ind_nifty200list20220101.csv`) both 404.

Backtesting today's constituents over history is therefore survivorship-biased
in two directions at once: names that fell out of the index (usually after
underperforming) are missing entirely, and names that entered did so partly
*because* they outperformed. Both push backtested returns up.

This module keeps a dated snapshot archive under `data_cache/universe/`, and
`get_universe(as_of=...)` returns the snapshot that was in effect on that date.
Every refresh writes a new dated snapshot, so the archive becomes progressively
more useful going forward. Until it spans the whole backtest window,
`coverage()` reports SURVIVORSHIP_BIASED and callers are expected to surface
that in their output rather than quietly reporting a clean-looking number.

This mirrors freqtrade's approach, where every pairlist filter declares its own
`supports_backtesting` so a biased universe can't be used by accident.
"""
import enum
import glob
import io
import os
from datetime import date

import pandas as pd
import requests

NIFTY200_URL = "https://archives.nseindia.com/content/indices/ind_nifty200list.csv"

_DIR = os.path.dirname(__file__)
SNAPSHOT_DIR = os.path.join(_DIR, "data_cache", "universe")
SNAPSHOT_PREFIX = "nifty200_"
# Pre-existing undated cache, kept in sync so older code paths keep working.
LEGACY_CACHE_PATH = os.path.join(_DIR, "data_cache", "nifty200.csv")


class UniverseCoverage(enum.StrEnum):
    """Whether the snapshot archive actually covers a requested backtest window."""

    POINT_IN_TIME = "point_in_time"
    SURVIVORSHIP_BIASED = "survivorship_biased"


def _snapshot_path(d: date) -> str:
    return os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_PREFIX}{d.isoformat()}.csv")


def _parse_snapshot_date(path: str) -> date | None:
    name = os.path.basename(path)
    if not (name.startswith(SNAPSHOT_PREFIX) and name.endswith(".csv")):
        return None
    try:
        return date.fromisoformat(name[len(SNAPSHOT_PREFIX) : -len(".csv")])
    except ValueError:
        return None


def _migrate_legacy_cache() -> None:
    """Seed the archive from the old undated cache, dated by its file mtime.

    Approximate, but strictly better than treating that file as valid for all
    of history -- and it only ever runs once, before the first real snapshot.
    """
    if not os.path.exists(LEGACY_CACHE_PATH):
        return
    if glob.glob(os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_PREFIX}*.csv")):
        return
    fetched = date.fromtimestamp(os.path.getmtime(LEGACY_CACHE_PATH))
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    pd.read_csv(LEGACY_CACHE_PATH).to_csv(_snapshot_path(fetched), index=False)


def available_snapshots() -> list[tuple[date, str]]:
    """All dated constituent snapshots on disk, oldest first."""
    _migrate_legacy_cache()
    out = []
    for path in glob.glob(os.path.join(SNAPSHOT_DIR, f"{SNAPSHOT_PREFIX}*.csv")):
        d = _parse_snapshot_date(path)
        if d is not None:
            out.append((d, path))
    return sorted(out)


def refresh_snapshot() -> list[str]:
    """Fetch the live list from NSE and record it as today's snapshot."""
    resp = requests.get(NIFTY200_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text))
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    df.to_csv(_snapshot_path(date.today()), index=False)
    df.to_csv(LEGACY_CACHE_PATH, index=False)
    return df["Symbol"].tolist()


def get_universe(as_of=None, refresh: bool = False) -> list[str]:
    """NSE trading symbols for the Nifty 200 as of a date (default: latest known).

    Falls back to the earliest available snapshot when `as_of` predates the
    archive -- that fallback is exactly the survivorship bias `coverage()`
    reports on.
    """
    if refresh:
        try:
            return refresh_snapshot()
        except Exception:
            if not available_snapshots():
                raise

    snapshots = available_snapshots()
    if not snapshots:
        return refresh_snapshot()

    if as_of is None:
        return _read_symbols(snapshots[-1][1])

    as_of_date = pd.Timestamp(as_of).date()
    effective = [(d, p) for d, p in snapshots if d <= as_of_date]
    chosen = effective[-1] if effective else snapshots[0]
    return _read_symbols(chosen[1])


def universe_union(symbols_by_date: list[tuple[date, str]] | None = None) -> list[str]:
    """Every symbol that appears in any snapshot.

    A backtest needs price history for the union, then restricts to
    point-in-time membership at each rebalance.
    """
    snapshots = symbols_by_date if symbols_by_date is not None else available_snapshots()
    seen: set[str] = set()
    for _, path in snapshots:
        seen.update(_read_symbols(path))
    return sorted(seen)


def coverage(start) -> UniverseCoverage:
    """Whether the archive has a snapshot at or before `start`."""
    snapshots = available_snapshots()
    if snapshots and snapshots[0][0] <= pd.Timestamp(start).date():
        return UniverseCoverage.POINT_IN_TIME
    return UniverseCoverage.SURVIVORSHIP_BIASED


def bias_warning(start) -> str | None:
    """Human-readable survivorship warning, or None when coverage is clean."""
    if coverage(start) is UniverseCoverage.POINT_IN_TIME:
        return None
    snapshots = available_snapshots()
    earliest = snapshots[0][0].isoformat() if snapshots else "no snapshots on disk"
    return (
        "SURVIVORSHIP BIAS: no constituent snapshot at or before "
        f"{pd.Timestamp(start).date()} (earliest available: {earliest}). "
        "Rebalances before that date use the earliest known membership, which "
        "excludes names later dropped from the index and includes names that "
        "had not yet joined. Reported CAGR/Sharpe are optimistic by an unknown "
        "margin. NSE publishes no dated archive; snapshots accumulate only from "
        "each refresh onward."
    )


def _read_symbols(path: str) -> list[str]:
    return pd.read_csv(path)["Symbol"].tolist()


def groww_symbol(trading_symbol: str, exchange: str = "NSE") -> str:
    return f"{exchange}-{trading_symbol}"


if __name__ == "__main__":
    snapshots = available_snapshots()
    print(f"{len(snapshots)} snapshot(s) on disk:")
    for d, path in snapshots:
        print(f"  {d}  {len(_read_symbols(path))} symbols  {os.path.basename(path)}")
    print(f"\nUnion across snapshots: {len(universe_union())} symbols")
    print(f"Coverage from 2022-01-01: {coverage('2022-01-01')}")
    warning = bias_warning("2022-01-01")
    if warning:
        print(f"\n{warning}")
