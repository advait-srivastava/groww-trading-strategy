"""Market-internal regime signal: breadth + index volatility, from Groww data only.

Complements strategy/macro.py (external USD/INR + US yield signal) with a
bottom-up read of the Nifty 200 universe itself:

  - Breadth: % of the universe trading above its own 200-day SMA. Low breadth
    means the "average stock" is in a downtrend even if the index looks fine
    (large-cap-driven rallies can mask broad weakness).
  - Index volatility: NIFTY 50's trailing 20-day realized vol vs its own
    trailing 1-year median. A spike means the ride is getting rougher even if
    price hasn't fallen much yet.

Both are cheap to compute from data strategy/data.py already fetches -- no
extra API calls or dependencies.
"""
import numpy as np
import pandas as pd

BREADTH_SMA_DAYS = 200
BREADTH_RISK_OFF_THRESHOLD = 0.40  # <40% of universe above its 200d SMA -> risk-off

VOL_WINDOW_DAYS = 20
VOL_LOOKBACK_DAYS = 252
VOL_SPIKE_MULTIPLE = 1.5  # 20d realized vol > 1.5x its trailing 1yr median -> risk-off

RISK_OFF_CASH_BUFFER = 0.25


def breadth_signal(price_panel: pd.DataFrame, as_of) -> dict:
    """Fraction of the universe trading above its own 200-day SMA as of a date."""
    prices = price_panel.loc[:as_of]
    if len(prices) < BREADTH_SMA_DAYS:
        return {"breadth": None, "risk_off": False}

    sma = prices.iloc[-BREADTH_SMA_DAYS:].mean()
    last = prices.iloc[-1]
    has_history = prices.iloc[-BREADTH_SMA_DAYS:].notna().all()
    above = (last > sma) & has_history
    eligible = has_history

    if eligible.sum() == 0:
        return {"breadth": None, "risk_off": False}

    breadth = above.sum() / eligible.sum()
    return {"breadth": breadth, "risk_off": breadth < BREADTH_RISK_OFF_THRESHOLD}


def volatility_signal(index_close: pd.Series, as_of) -> dict:
    """NIFTY 50's trailing 20d realized vol vs its trailing 1yr median."""
    closes = index_close.loc[:as_of].dropna()
    if len(closes) < VOL_LOOKBACK_DAYS + VOL_WINDOW_DAYS:
        return {"realized_vol": None, "median_vol": None, "risk_off": False}

    returns = closes.pct_change().dropna()
    rolling_vol = returns.rolling(VOL_WINDOW_DAYS).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    current = rolling_vol.iloc[-1]
    median = rolling_vol.iloc[-VOL_LOOKBACK_DAYS:].median()
    risk_off = median > 0 and current > median * VOL_SPIKE_MULTIPLE
    return {"realized_vol": current, "median_vol": median, "risk_off": risk_off}


def regime_signal(price_panel: pd.DataFrame, index_close: pd.Series, as_of) -> dict:
    """Combined breadth + volatility read as of a given date."""
    breadth = breadth_signal(price_panel, as_of)
    vol = volatility_signal(index_close, as_of)
    return {
        "as_of": as_of,
        "breadth": breadth["breadth"],
        "realized_vol": vol["realized_vol"],
        "median_vol": vol["median_vol"],
        "risk_off": breadth["risk_off"] or vol["risk_off"],
        "breadth_risk_off": breadth["risk_off"],
        "vol_risk_off": vol["risk_off"],
    }


def cash_buffer(price_panel: pd.DataFrame, index_close: pd.Series, as_of, normal: float = 0.05) -> float:
    return RISK_OFF_CASH_BUFFER if regime_signal(price_panel, index_close, as_of)["risk_off"] else normal


if __name__ == "__main__":
    from datetime import datetime
    from strategy import factors
    from strategy.data import fetch_universe_history, fetch_history
    from strategy.universe import get_universe

    end = datetime.today()
    start = datetime(2020, 1, 1)
    symbols = get_universe()
    history = fetch_universe_history(symbols, start, end)
    price_panel = factors.build_close_panel(history)
    nifty = fetch_history("NIFTY", start, end)["close"]

    signal = regime_signal(price_panel, nifty, price_panel.index.max())
    print(signal)
