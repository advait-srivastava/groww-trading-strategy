"""Rules for the weekly NIFTY bull put credit spread (shared by backtest + live).

Strategy, in plain terms:
  1. Only trade when the internal regime signal (strategy/regime.py) reads
     risk-on -- selling puts into a weakening market is exactly the wrong
     time to do it.
  2. Sell a put near SHORT_PUT_DELTA_TARGET delta (a "high probability,
     low premium" strike -- roughly an 18% chance of finishing ITM at
     entry), buy a further-OTM put SPREAD_WIDTH_POINTS below it to cap risk.
  3. Close early at PROFIT_TARGET_PCT of the credit received, or at
     STOP_LOSS_MULTIPLE x the credit received if it moves against us --
     never ride a spread into its last day or two, where gamma is worst
     (see the gamma discussion -- pin risk right at the short strike is
     exactly what this avoids).
  4. Otherwise let it expire (NIFTY options are cash-settled -- no
     assignment/delivery risk either way).

One position at a time, sized to fit within OPTIONS_BUDGET.
"""
import numpy as np
import pandas as pd

SHORT_PUT_DELTA_TARGET = -0.18
SPREAD_WIDTH_POINTS = 100.0  # matches the width we verified fits the Rs 50K budget via real margin check
STRIKE_STEP = 50.0
LOT_SIZE = 65  # current NIFTY lot size; historical lot sizes differed (another backtest approximation)

PROFIT_TARGET_PCT = 0.50  # close at 50% of max credit captured
# No active stop-loss: the long put already caps max loss at spread width, and
# backtesting showed every stop threshold tested (1.5x/2x/4x credit) made
# results WORSE than none -- a stop just locks in losses on trades that would
# have mean-reverted back to profitable. The defined-risk structure IS the
# risk control here. Kept as a disaster-only backstop, not an active exit.
STOP_LOSS_MULTIPLE = 10.0

IV_LOOKBACK_DAYS = 20
TRADING_DAYS_PER_YEAR = 252

OPTIONS_BUDGET = 50_000.0


def realized_vol(price_series: pd.Series, as_of, lookback: int = IV_LOOKBACK_DAYS) -> float | None:
    """Annualized realized volatility over the trailing `lookback` days -- our IV proxy."""
    prices = price_series.loc[:as_of].dropna()
    if len(prices) < lookback + 1:
        return None
    returns = prices.iloc[-lookback - 1:].pct_change().dropna()
    return float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def select_bull_put_strikes(spot: float, T_years: float, sigma: float) -> dict:
    from strategy.options_pricing import strike_for_put_delta

    short_strike = strike_for_put_delta(
        spot, T_years, sigma, SHORT_PUT_DELTA_TARGET, strike_step=STRIKE_STEP
    )
    long_strike = short_strike - SPREAD_WIDTH_POINTS
    return {"short_strike": short_strike, "long_strike": long_strike}


def should_trade(regime_signal: dict | None) -> bool:
    """Only sell premium when the internal regime reads risk-on."""
    if regime_signal is None:
        return False
    return not regime_signal.get("risk_off", True)


def spread_payoff(short_strike: float, long_strike: float, settlement_price: float) -> float:
    """What the spread SELLER owes at expiry (>=0), per unit of underlying."""
    short_put_intrinsic = max(short_strike - settlement_price, 0.0)
    long_put_intrinsic = max(long_strike - settlement_price, 0.0)
    return short_put_intrinsic - long_put_intrinsic
