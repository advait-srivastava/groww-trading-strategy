"""Black-Scholes pricing for the synthetic options backtest.

Groww's instrument master only carries currently-live contracts -- expired
option chains are unrecoverable, so there's no way to backtest against real
historical option prices (confirmed by checking get_all_instruments(): the
earliest listed NIFTY option expiry is the *next* one, nothing historical).

This reconstructs theoretical European option prices from real historical
NIFTY closes using Black-Scholes with realized volatility as an IV proxy.
That's a real approximation, not equivalent to backtesting on actual traded
prices -- see the caveats in strategy/options_backtest.py's docstring.
"""
import numpy as np
from scipy.stats import norm

RISK_FREE_RATE = 0.065  # approx Indian T-bill yield


def _d1_d2(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bs_put_price(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    if T <= 0:
        return max(K - S, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    if T <= 0:
        return max(S - K, 0.0)
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_delta(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    if T <= 0:
        return -1.0 if S < K else 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return norm.cdf(d1) - 1.0


def bs_call_delta(S: float, K: float, T: float, sigma: float, r: float = RISK_FREE_RATE) -> float:
    if T <= 0:
        return 1.0 if S > K else 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return norm.cdf(d1)


def strike_for_put_delta(
    S: float, T: float, sigma: float, target_delta: float, r: float = RISK_FREE_RATE, strike_step: float = 50.0
) -> float:
    """Nearest strike (rounded to strike_step) whose put delta is closest to target_delta (negative)."""
    strikes = np.arange(S * 0.80, S * 1.02, strike_step)
    deltas = np.array([bs_put_delta(S, k, T, sigma, r) for k in strikes])
    idx = np.argmin(np.abs(deltas - target_delta))
    return float(strikes[idx])
