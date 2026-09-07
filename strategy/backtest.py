"""Backtest the momentum + trend-filter strategy over Nifty 200 history.

Strategy (positional, monthly rebalance):
  1. Universe: Nifty 200, filtered for liquidity (>= Rs 5cr/day avg turnover).
  2. Trend filter: only consider stocks trading above their 200-day SMA
     (absolute momentum / regime filter -- cuts exposure in broad downtrends).
  3. Rank survivors by 12-1 month momentum (12mo return, skipping the most
     recent month to avoid short-term reversal).
  4. Hold the top N, inverse-volatility weighted, capped per-name.
  5. Rebalance monthly. Transaction costs applied on turnover.

This mirrors strategy/factors.py exactly, which is also what the live
rebalance script uses -- so backtest results reflect what live trading
would actually do.
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from strategy import data as data_quality
from strategy import factors, macro, regime, universe
from strategy.data import fetch_history, fetch_universe_history
from strategy.universe import get_universe

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.065  # approx Indian T-bill yield, for Sharpe


def rebalance_dates(index: pd.DatetimeIndex, start: pd.Timestamp) -> list[pd.Timestamp]:
    months = pd.date_range(start, index.max(), freq="MS")
    dates = []
    for m in months:
        candidates = index[index >= m]
        if len(candidates):
            dates.append(candidates[0])
    return sorted(set(dates))


def run_backtest(
    price_panel: pd.DataFrame,
    turnover_panel: pd.DataFrame,
    backtest_start: pd.Timestamp,
    initial_capital: float = 10_00_000.0,
    top_n: int = 18,
    max_weight: float = 0.12,
    cash_buffer: float = 0.05,
    cost_bps: float = 10.0,
    use_macro_regime: bool = False,
    use_internal_regime: bool = True,
    index_close: pd.Series | None = None,
    exec_panel: pd.DataFrame | None = None,
    mark_panel: pd.DataFrame | None = None,
    universe_at=None,
) -> dict:
    """Simulate the strategy.

    Execution model: a rebalance decided from day N's close is executed on day
    N+1, one full session later. Deciding and transacting on the same close
    would guarantee filling at exactly the price the signal was derived from,
    which is a lookahead the live script cannot reproduce.

    `exec_panel` supplies the fill prices for that next session -- defaults to
    `price_panel` (next close). Passing an open panel gives next-open fills,
    but only do that once the open data is verified real; see
    data.open_price_quality(). `mark_panel` (fully forward-filled) values
    existing holdings so a data gap doesn't mark a position to zero. Held names
    that go stale in `price_panel` are force-exited at their last known mark
    rather than carried indefinitely.

    `universe_at(day) -> list[str]` supplies point-in-time index membership.
    """
    dates = price_panel.index
    rdates = rebalance_dates(dates, backtest_start)
    if not rdates:
        raise ValueError("No rebalance dates in range -- check backtest_start vs data history")

    exec_panel = exec_panel if exec_panel is not None else price_panel
    marks = mark_panel if mark_panel is not None else price_panel

    shares = pd.Series(0.0, index=price_panel.columns)
    cash = initial_capital
    equity_curve = []
    turnover_log = []
    holdings_log = []
    stale_exits = []

    daily_index = dates[(dates >= rdates[0]) & (dates <= dates.max())]
    rdate_set = set(rdates)
    pending = None  # signal awaiting execution at the next bar's open

    for day in daily_index:
        marks_today = marks.loc[day]

        # 1. Force-exit holdings with no recent tradeable price (suspension /
        #    delisting), at the last known mark.
        held = shares[shares > 0].index
        if len(held):
            stale = held[price_panel.loc[day, held].isna()]
            if len(stale):
                proceeds = float((shares[stale] * marks_today[stale].fillna(0)).sum())
                cash += proceeds
                shares[stale] = 0.0
                stale_exits.append({
                    "date": day,
                    "symbols": ",".join(stale),
                    "n_symbols": len(stale),
                    "proceeds": proceeds,
                })

        # 2. Execute the previous session's decision at today's open.
        if pending is not None:
            weights, effective_buffer, flags = pending
            pending = None

            exec_prices = exec_panel.loc[day].replace(0, np.nan)
            untradeable = exec_prices.isna()
            # Value everything at execution prices where they exist, last mark
            # otherwise, so the cash identity below conserves value exactly.
            valuation = exec_prices.fillna(marks_today).fillna(0)

            portfolio_value = cash + float((shares * valuation).sum())
            investable = portfolio_value * (1 - effective_buffer)

            target_value = pd.Series(0.0, index=price_panel.columns)
            if not weights.empty:
                target_value.loc[weights.index] = investable * weights

            new_shares = (target_value / exec_prices).fillna(0.0)
            new_shares[untradeable] = shares[untradeable]  # can't trade, hold as-is

            traded = ~untradeable
            gross_turnover = float(
                ((new_shares[traded] - shares[traded]) * valuation[traded]).abs().sum()
            )
            cost = gross_turnover * (cost_bps / 10_000.0)
            cash = portfolio_value - float((new_shares * valuation).sum()) - cost

            shares = new_shares
            turnover_log.append({"date": day, "gross_turnover": gross_turnover, "cost": cost})
            holdings_log.append({
                "date": day,
                "signal_date": flags["signal_date"],
                "n_positions": int((shares > 0).sum()),
                "n_untradeable": int((untradeable & (target_value > 0)).sum()),
                "cash_buffer": effective_buffer,
                "risk_off": flags["risk_off"],
                "macro_risk_off": flags["macro_risk_off"],
                "internal_risk_off": flags["internal_risk_off"],
            })

        # 3. Generate a signal from today's close, for execution next session.
        if day in rdate_set:
            weights = factors.select_portfolio(
                price_panel,
                turnover_panel,
                day,
                top_n=top_n,
                max_weight=max_weight,
                allowed=universe_at(day) if universe_at is not None else None,
            )

            macro_risk_off = False
            if use_macro_regime:
                macro_risk_off = macro.regime_signal(day)["risk_off"]

            internal_risk_off = False
            if use_internal_regime and index_close is not None:
                internal_risk_off = regime.regime_signal(price_panel, index_close, day)["risk_off"]

            risk_off = macro_risk_off or internal_risk_off
            effective_buffer = macro.RISK_OFF_CASH_BUFFER if risk_off else cash_buffer
            pending = (
                weights,
                effective_buffer,
                {
                    "signal_date": day,
                    "risk_off": risk_off,
                    "macro_risk_off": macro_risk_off,
                    "internal_risk_off": internal_risk_off,
                },
            )

        mtm = cash + float((shares * marks_today.fillna(0)).sum())
        equity_curve.append({"date": day, "equity": mtm})

    unexecuted = pending[2]["signal_date"] if pending is not None else None

    equity_df = pd.DataFrame(equity_curve).set_index("date")["equity"]
    turnover_df = pd.DataFrame(turnover_log).set_index("date") if turnover_log else pd.DataFrame()
    holdings_df = pd.DataFrame(holdings_log).set_index("date") if holdings_log else pd.DataFrame()
    stale_df = pd.DataFrame(stale_exits).set_index("date") if stale_exits else pd.DataFrame()

    return {
        "equity_curve": equity_df,
        "turnover": turnover_df,
        "holdings": holdings_df,
        "stale_exits": stale_df,
        "rebalance_dates": rdates,
        "unexecuted_signal": unexecuted,
    }


def benchmark_curve(price_panel: pd.DataFrame, symbol: str, backtest_start: pd.Timestamp, initial_capital: float) -> pd.Series:
    series = price_panel[symbol].dropna()
    series = series[series.index >= backtest_start]
    shares = initial_capital / series.iloc[0]
    return series * shares


def performance_stats(equity: pd.Series, label: str) -> dict:
    returns = equity.pct_change().dropna()
    n_years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else np.nan
    ann_vol = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = (cagr - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan

    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    return {
        "label": label,
        "start": equity.index[0].date(),
        "end": equity.index[-1].date(),
        "start_value": equity.iloc[0],
        "end_value": equity.iloc[-1],
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "calmar": calmar,
    }


def print_stats(stats: dict):
    print(f"\n--- {stats['label']} ---")
    print(f"  Period:          {stats['start']} to {stats['end']}")
    print(f"  Start -> End:    Rs {stats['start_value']:,.0f} -> Rs {stats['end_value']:,.0f}")
    print(f"  CAGR:            {stats['cagr']*100:.2f}%")
    print(f"  Annualized vol:  {stats['annualized_vol']*100:.2f}%")
    print(f"  Sharpe (rf 6.5%):{stats['sharpe']:.2f}")
    print(f"  Max drawdown:    {stats['max_drawdown']*100:.2f}%")
    print(f"  Calmar ratio:    {stats['calmar']:.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-start", default="2020-01-01")
    parser.add_argument("--backtest-start", default="2022-01-01")
    parser.add_argument("--end", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--capital", type=float, default=10_00_000.0)
    parser.add_argument("--top-n", type=int, default=18)
    parser.add_argument("--max-weight", type=float, default=0.12)
    parser.add_argument("--cash-buffer", type=float, default=0.05)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--macro-regime",
        action="store_true",
        help="Enable the Alpha Vantage macro overlay (USD/INR + US 10Y yield). "
        "Off by default: backtested worse than doing nothing (used alone, CAGR 23.39%% vs "
        "25.49%%, Sharpe 0.82 vs 0.89) -- see strategy/regime.py vs strategy/macro.py.",
    )
    parser.add_argument(
        "--no-internal-regime",
        action="store_true",
        help="Disable the internal breadth + volatility regime overlay (Groww data only). "
        "On by default, but for drawdown rather than Sharpe: it costs 1.3 points of CAGR "
        "(25.49%%->24.18%%) and leaves Sharpe flat (0.89->0.88) while cutting max drawdown "
        "-25.27%%->-21.05%% and lifting Calmar 1.01->1.15.",
    )
    parser.add_argument(
        "--exec-price",
        choices=["next_close", "next_open"],
        default="next_close",
        help="Fill price for the session after a signal. Default next_close: the cached "
        "open column is synthetic (open == previous close) for 2025+, so next_open "
        "would silently collapse into same-bar execution. next_open is validated at "
        "runtime and falls back if the data is untrustworthy.",
    )
    parser.add_argument(
        "--point-in-time",
        action="store_true",
        help="Restrict each rebalance to the Nifty 200 members as of that date, using the "
        "dated snapshot archive in strategy/data_cache/universe/. Off by default only "
        "because the archive does not yet span the backtest window -- see "
        "strategy/universe.py for why NSE makes this hard.",
    )
    parser.add_argument("--out", default="strategy/data_cache/equity_curve.csv")
    args = parser.parse_args()

    data_start = datetime.strptime(args.data_start, "%Y-%m-%d")
    backtest_start = pd.Timestamp(args.backtest_start)
    end = datetime.strptime(args.end, "%Y-%m-%d")

    # Price history is loaded for the union of every constituent snapshot; the
    # backtest then restricts each rebalance to point-in-time membership.
    symbols = universe.universe_union() if args.point_in_time else get_universe()
    print(f"Loading cached history for {len(symbols)} symbols...")
    history = fetch_universe_history(symbols, data_start, end)
    print(f"Loaded {len(history)}/{len(symbols)} symbols with data.")

    coverage = universe.coverage(backtest_start)
    print(f"Universe mode: {'point-in-time' if args.point_in_time else 'current-constituents'}"
          f"  (coverage: {coverage})")
    warning = universe.bias_warning(backtest_start)
    if warning:
        print(f"\n!! {warning}\n")

    nifty = fetch_history("NIFTY", data_start, end)

    price_panel = factors.build_close_panel(history)
    mark_panel = factors.build_mark_panel(history)
    turnover_panel = factors.build_turnover_panel(history)

    exec_panel = None  # None -> fill at the next session's close
    if args.exec_price == "next_open":
        quality = data_quality.open_price_quality(history)
        suspect = data_quality.suspect_open_years(quality)
        print(
            f"Open-price check: {quality['synthetic_share']*100:.1f}% of "
            f"{quality['rows']:,} rows have open == previous close"
        )
        if suspect:
            detail = ", ".join(f"{y} {share*100:.1f}% of {n:,}" for y, n, share in suspect)
            print(
                f"!! Refusing --exec-price next_open: open == previous close in {detail}. "
                "Those opens are synthetic, so next-open fills would collapse back into "
                "same-bar execution. Falling back to next_close."
            )
        else:
            exec_panel = factors.build_open_panel(history)

    universe_at = None
    if args.point_in_time:
        cache: dict[pd.Timestamp, list[str]] = {}

        def universe_at(day):  # noqa: F811 -- deliberate conditional definition
            if day not in cache:
                cache[day] = get_universe(as_of=day)
            return cache[day]

    result = run_backtest(
        price_panel,
        turnover_panel,
        backtest_start,
        initial_capital=args.capital,
        top_n=args.top_n,
        max_weight=args.max_weight,
        cash_buffer=args.cash_buffer,
        cost_bps=args.cost_bps,
        use_macro_regime=args.macro_regime,
        use_internal_regime=not args.no_internal_regime,
        index_close=nifty["close"],
        exec_panel=exec_panel,
        mark_panel=mark_panel,
        universe_at=universe_at,
    )

    strat_stats = performance_stats(result["equity_curve"], "Momentum + Trend Strategy")
    print_stats(strat_stats)

    bench = benchmark_curve(pd.DataFrame({"NIFTY": nifty["close"]}), "NIFTY", backtest_start, args.capital)
    bench_stats = performance_stats(bench, "NIFTY 50 Buy & Hold (benchmark)")
    print_stats(bench_stats)

    avg_turnover = result["turnover"]["gross_turnover"].mean() if not result["turnover"].empty else 0
    total_cost = result["turnover"]["cost"].sum() if not result["turnover"].empty else 0
    avg_positions = result["holdings"]["n_positions"].mean() if not result["holdings"].empty else 0
    print(f"\n--- Trading activity ---")
    print(f"  Rebalances:          {len(result['rebalance_dates'])}")
    effective_exec = "next close" if exec_panel is None else "next open"
    print(f"  Executed:            {len(result['turnover'])} "
          f"(signal at close, filled next session @ {effective_exec})")
    print(f"  Avg positions held:  {avg_positions:.1f}")
    print(f"  Avg turnover/rebal:  Rs {avg_turnover:,.0f}")
    print(f"  Total cost drag:     Rs {total_cost:,.0f}")
    if result["unexecuted_signal"] is not None:
        print(f"  Unexecuted signal:   {result['unexecuted_signal'].date()} (no next session in data)")
    stale = result["stale_exits"]
    if not stale.empty:
        print(f"  Stale-price exits:   {int(stale['n_symbols'].sum())} position(s) "
              f"on {len(stale)} day(s), Rs {stale['proceeds'].sum():,.0f} recovered")
    if not result["holdings"].empty and result["holdings"]["n_untradeable"].sum():
        print(f"  Skipped (no open):   {int(result['holdings']['n_untradeable'].sum())} target position(s)")
    if not result["holdings"].empty and (args.macro_regime or not args.no_internal_regime):
        h = result["holdings"]
        print(f"  Risk-off rebalances: {int(h['risk_off'].sum())}/{len(result['rebalance_dates'])} (combined)")
        if args.macro_regime:
            print(f"    - macro (AV) risk-off:    {int(h['macro_risk_off'].sum())}")
        if not args.no_internal_regime:
            print(f"    - internal (breadth/vol): {int(h['internal_risk_off'].sum())}")

    combined = pd.DataFrame({
        "strategy": result["equity_curve"],
        "nifty50": bench.reindex(result["equity_curve"].index).ffill(),
    })
    combined.to_csv(args.out)
    print(f"\nEquity curve saved to {args.out}")


if __name__ == "__main__":
    main()
