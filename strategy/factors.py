"""Shared signal logic: trend filter, momentum ranking, volatility weighting.

Used identically by the backtester and the live rebalance script, so a
backtested result reflects exactly what the live script would have done.
"""
import numpy as np
import pandas as pd

TREND_SMA_DAYS = 200
MOMENTUM_LOOKBACK_DAYS = 252  # ~12 months
MOMENTUM_SKIP_DAYS = 21  # skip most recent ~1 month (avoids short-term reversal)
VOL_LOOKBACK_DAYS = 20
LIQUIDITY_LOOKBACK_DAYS = 20
MIN_AVG_DAILY_TURNOVER = 5_00_00_000  # Rs 5 crore/day, keeps slippage low


def build_close_panel(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Wide close-price panel (dates x symbols), forward-filled for holidays/listings gaps."""
    closes = {sym: df["close"] for sym, df in history.items()}
    panel = pd.DataFrame(closes).sort_index()
    return panel.ffill()


def build_turnover_panel(history: dict[str, pd.DataFrame]) -> pd.DataFrame:
    turnover = {sym: df["close"] * df["volume"] for sym, df in history.items()}
    return pd.DataFrame(turnover).sort_index().ffill()


def eligible_mask(price_panel: pd.DataFrame, turnover_panel: pd.DataFrame, as_of) -> pd.Series:
    """Stocks that pass the trend + liquidity filter as of a given date."""
    prices = price_panel.loc[:as_of]
    if len(prices) < TREND_SMA_DAYS:
        return pd.Series(False, index=price_panel.columns)

    sma200 = prices.iloc[-TREND_SMA_DAYS:].mean()
    last_price = prices.iloc[-1]
    trend_ok = last_price > sma200

    turnover = turnover_panel.loc[:as_of]
    liquidity_ok = turnover.iloc[-LIQUIDITY_LOOKBACK_DAYS:].mean() >= MIN_AVG_DAILY_TURNOVER

    has_history = prices.iloc[-TREND_SMA_DAYS:].notna().all()

    return trend_ok & liquidity_ok & has_history


def momentum_score(price_panel: pd.DataFrame, as_of) -> pd.Series:
    """12-1 month momentum: return from 12mo ago to 1mo ago, skipping the recent month."""
    prices = price_panel.loc[:as_of]
    if len(prices) < MOMENTUM_LOOKBACK_DAYS + 1:
        return pd.Series(np.nan, index=price_panel.columns)

    p_recent = prices.iloc[-1 - MOMENTUM_SKIP_DAYS]
    p_past = prices.iloc[-1 - MOMENTUM_LOOKBACK_DAYS]
    return (p_recent / p_past) - 1.0


def inverse_vol_weights(
    price_panel: pd.DataFrame, as_of, symbols: list[str], max_weight: float = 0.12
) -> pd.Series:
    """Inverse trailing-volatility weights over the given symbols, capped and renormalized."""
    prices = price_panel.loc[:as_of, symbols]
    returns = prices.iloc[-VOL_LOOKBACK_DAYS:].pct_change().dropna(how="all")
    vol = returns.std()
    vol = vol.replace(0, np.nan)
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.fillna(inv_vol.min())
    weights = inv_vol / inv_vol.sum()

    # Iteratively cap and redistribute so no single name exceeds max_weight.
    for _ in range(10):
        over = weights > max_weight
        if not over.any():
            break
        excess = (weights[over] - max_weight).sum()
        weights[over] = max_weight
        under = ~over
        if weights[under].sum() == 0:
            break
        weights[under] += excess * (weights[under] / weights[under].sum())

    return weights


def select_portfolio(
    price_panel: pd.DataFrame,
    turnover_panel: pd.DataFrame,
    as_of,
    top_n: int = 18,
    max_weight: float = 0.12,
) -> pd.Series:
    """Full pipeline: filter -> rank by momentum -> pick top N -> inverse-vol weight."""
    eligible = eligible_mask(price_panel, turnover_panel, as_of)
    eligible_symbols = eligible[eligible].index

    if len(eligible_symbols) == 0:
        return pd.Series(dtype=float)

    scores = momentum_score(price_panel, as_of).loc[eligible_symbols].dropna()
    scores = scores[scores > 0]  # require positive momentum, not just "least bad"
    if scores.empty:
        return pd.Series(dtype=float)

    picks = scores.sort_values(ascending=False).head(top_n).index.tolist()
    return inverse_vol_weights(price_panel, as_of, picks, max_weight=max_weight)
