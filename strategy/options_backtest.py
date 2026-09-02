"""Synthetic backtest for the weekly NIFTY bull put credit spread.

IMPORTANT CAVEAT: Groww's instrument master only carries currently-live
option contracts -- expired chains are unrecoverable (verified directly:
get_all_instruments() shows the earliest listed NIFTY option expiry is
*tomorrow's*, nothing historical). There is no way to backtest this
strategy against real traded option prices, unlike strategy/backtest.py
for the equity strategy.

What this does instead: reconstruct theoretical option prices from real
historical NIFTY closes (strategy/data.py, actual Groww data) using
Black-Scholes with trailing realized volatility as an IV proxy
(strategy/options_pricing.py). This is a legitimate, commonly-used
approximation -- but it is NOT equivalent to the equity backtest's rigor.
Known gaps:
  - Realized vol != implied vol. IV usually sits above realized vol (the
    "volatility risk premium"), so this likely UNDERSTATES real option
    premium income -- a conservative bias, but a real one.
  - No bid-ask spread or liquidity effects on far-OTM strikes.
  - Margin is approximated from a single real data point (see
    strategy/options_strategy.py), not computed exactly per trade.
  - Uses today's NIFTY lot size (65) throughout; historical lot sizes
    differed.
  - "Weekly" cycles are simulated as fixed 5-trading-day windows, not the
    actual historical NSE expiry calendar (which has changed over time).

Treat this as a directional sanity check, not proof. Forward paper-tracking
via strategy/options_rebalance.py (dry-run) is the real validation step.
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from strategy import factors, options_strategy as ostrat, regime
from strategy.data import fetch_history, fetch_universe_history
from strategy.options_pricing import bs_put_price
from strategy.universe import get_universe

WEEK_TRADING_DAYS = 5
TRADING_DAYS_PER_YEAR = 252


def run_options_backtest(
    nifty_close: pd.Series,
    price_panel: pd.DataFrame,
    backtest_start: pd.Timestamp,
    budget: float = ostrat.OPTIONS_BUDGET,
) -> dict:
    dates = nifty_close.loc[backtest_start:].index
    entry_dates = dates[::WEEK_TRADING_DAYS]

    trades = []
    equity = budget
    equity_curve = [{"date": dates[0], "equity": equity}]

    for entry_date in entry_dates:
        idx = dates.get_loc(entry_date)
        exit_idx = min(idx + WEEK_TRADING_DAYS, len(dates) - 1)
        exit_date = dates[exit_idx]
        week_dates = dates[idx:exit_idx + 1]

        sig = regime.regime_signal(price_panel, nifty_close, entry_date)
        spot_entry = nifty_close.loc[entry_date]
        sigma = ostrat.realized_vol(nifty_close, entry_date)

        if not ostrat.should_trade(sig) or sigma is None:
            equity_curve.append({"date": exit_date, "equity": equity})
            trades.append({
                "entry_date": entry_date, "exit_date": exit_date, "traded": False,
                "reason": "risk-off" if sigma is not None else "insufficient history",
            })
            continue

        T_entry = WEEK_TRADING_DAYS / TRADING_DAYS_PER_YEAR
        strikes = ostrat.select_bull_put_strikes(spot_entry, T_entry, sigma)
        short_k, long_k = strikes["short_strike"], strikes["long_strike"]

        credit = bs_put_price(spot_entry, short_k, T_entry, sigma) - bs_put_price(spot_entry, long_k, T_entry, sigma)
        credit = max(credit, 0.01)
        max_loss = (short_k - long_k) - credit

        exit_reason = "expiry"
        pnl_per_unit = None
        for i, d in enumerate(week_dates[1:], start=1):
            days_left = WEEK_TRADING_DAYS - i
            T_remaining = max(days_left, 0.5) / TRADING_DAYS_PER_YEAR
            spot_t = nifty_close.loc[d]
            sigma_t = ostrat.realized_vol(nifty_close, d) or sigma
            spread_value = (
                bs_put_price(spot_t, short_k, T_remaining, sigma_t)
                - bs_put_price(spot_t, long_k, T_remaining, sigma_t)
            )
            profit_so_far = credit - spread_value
            if profit_so_far >= ostrat.PROFIT_TARGET_PCT * credit:
                pnl_per_unit = profit_so_far
                exit_reason = "profit target"
                exit_date = d
                break
            if spread_value >= ostrat.STOP_LOSS_MULTIPLE * credit:
                pnl_per_unit = credit - spread_value
                exit_reason = "stop loss"
                exit_date = d
                break

        if pnl_per_unit is None:
            settlement = nifty_close.loc[week_dates[-1]]
            payoff_owed = ostrat.spread_payoff(short_k, long_k, settlement)
            pnl_per_unit = credit - payoff_owed
            exit_date = week_dates[-1]

        pnl_per_unit = max(pnl_per_unit, -max_loss)
        pnl = pnl_per_unit * ostrat.LOT_SIZE
        equity += pnl

        trades.append({
            "entry_date": entry_date, "exit_date": exit_date, "traded": True,
            "short_strike": short_k, "long_strike": long_k, "credit_per_unit": credit,
            "pnl": pnl, "exit_reason": exit_reason,
        })
        equity_curve.append({"date": exit_date, "equity": equity})

    equity_df = pd.DataFrame(equity_curve).drop_duplicates(subset="date").set_index("date")["equity"]
    trades_df = pd.DataFrame(trades)
    return {"equity_curve": equity_df, "trades": trades_df}


def print_report(result: dict, budget: float):
    trades = result["trades"]
    traded = trades[trades["traded"]]
    equity = result["equity_curve"]

    final = equity.iloc[-1]
    total_return = final / budget - 1
    n_weeks = len(trades)
    n_traded = len(traded)
    win_rate = (traded["pnl"] > 0).mean() if len(traded) else float("nan")

    running_max = equity.cummax()
    dd = (equity / running_max - 1).min()

    print(f"\n--- Weekly NIFTY Bull Put Spread (synthetic backtest) ---")
    print(f"  Period:            {equity.index[0].date()} to {equity.index[-1].date()}")
    print(f"  Budget:            Rs {budget:,.0f}")
    print(f"  Final equity:      Rs {final:,.0f}")
    print(f"  Total return:      {total_return*100:.2f}%")
    print(f"  Weeks:             {n_weeks} ({n_traded} traded, {n_weeks - n_traded} skipped risk-off)")
    print(f"  Win rate:          {win_rate*100:.1f}%")
    print(f"  Max drawdown:      {dd*100:.2f}%")
    if n_traded:
        print(f"  Avg credit/trade:  Rs {(traded['credit_per_unit'] * ostrat.LOT_SIZE).mean():,.0f}")
        print(f"  Avg P&L/trade:     Rs {traded['pnl'].mean():,.0f}")
        print(f"  Exit reasons:      {traded['exit_reason'].value_counts().to_dict()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-start", default="2020-01-01")
    parser.add_argument("--backtest-start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--budget", type=float, default=ostrat.OPTIONS_BUDGET)
    parser.add_argument("--delta-target", type=float, default=ostrat.SHORT_PUT_DELTA_TARGET)
    parser.add_argument("--width", type=float, default=ostrat.SPREAD_WIDTH_POINTS)
    parser.add_argument("--profit-target-pct", type=float, default=ostrat.PROFIT_TARGET_PCT)
    parser.add_argument("--stop-loss-multiple", type=float, default=ostrat.STOP_LOSS_MULTIPLE)
    parser.add_argument("--no-stop-loss", action="store_true", help="Disable the stop loss (let ride to profit target or expiry).")
    args = parser.parse_args()

    ostrat.SHORT_PUT_DELTA_TARGET = args.delta_target
    ostrat.SPREAD_WIDTH_POINTS = args.width
    ostrat.PROFIT_TARGET_PCT = args.profit_target_pct
    ostrat.STOP_LOSS_MULTIPLE = args.stop_loss_multiple if not args.no_stop_loss else 1e9

    data_start = datetime.strptime(args.data_start, "%Y-%m-%d")
    backtest_start = pd.Timestamp(args.backtest_start)
    end = datetime.strptime(args.end, "%Y-%m-%d")

    symbols = get_universe()
    print(f"Loading cached history for {len(symbols)} symbols (for the regime signal)...")
    history = fetch_universe_history(symbols, data_start, end)
    price_panel = factors.build_close_panel(history)
    nifty_close = fetch_history("NIFTY", data_start, end)["close"]

    result = run_options_backtest(nifty_close, price_panel, backtest_start, budget=args.budget)
    print_report(result, args.budget)
    print(
        "\nReminder: this is a Black-Scholes approximation on real NIFTY prices, "
        "not a backtest on real option prices (see this file's docstring for why)."
    )


if __name__ == "__main__":
    main()
