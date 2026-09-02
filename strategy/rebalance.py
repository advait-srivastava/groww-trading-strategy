"""Compute and (optionally) execute the monthly momentum/trend rebalance.

Uses the exact same signal logic as strategy/backtest.py (strategy/factors.py)
so live trades match what was backtested. Defaults to a dry run that only
prints the planned orders -- pass --confirm to actually place them.

Example:
    python -m strategy.rebalance                 # dry run
    python -m strategy.rebalance --confirm        # places real orders
"""
import argparse
import csv
import os
from datetime import datetime, timedelta

import pandas as pd
from growwapi import GrowwAPI

from client import get_client
from strategy import factors, macro, regime
from strategy.data import fetch_universe_history, fetch_history
from strategy.universe import get_universe

LOOKBACK_CALENDAR_DAYS = 800  # >2yr, covers 200d SMA + 12mo momentum with buffer
MIN_ORDER_VALUE = 500.0  # skip dust trades
TRADE_LOG = os.path.join(os.path.dirname(__file__), "data_cache", "trade_log.csv")


def get_live_prices(groww, symbols: list[str]) -> dict[str, float]:
    if not symbols:
        return {}
    pairs = tuple(f"NSE_{s}" for s in symbols)
    resp = groww.get_ltp(exchange_trading_symbols=pairs, segment=GrowwAPI.SEGMENT_CASH)
    return {sym: resp.get(f"NSE_{sym}") for sym in symbols}


def resolve_cash_buffer(
    price_panel, index_close, as_of, cash_buffer: float, use_internal_regime: bool, use_macro_regime: bool
) -> tuple[float, dict | None]:
    """Cash buffer to use, elevated when either regime overlay reads risk-off.

    Internal (breadth + volatility, Groww data) is on by default -- backtested
    to improve Sharpe 0.90->0.94 and cut max drawdown -24.85%->-20.97%. Macro
    (Alpha Vantage USD/INR + US 10Y yield) is off by default -- backtested
    worse than doing nothing (see strategy/backtest.py comparison). Falls back
    to the fixed cash_buffer (with a warning) if a signal can't be read, so a
    stale/unreachable data source never blocks a live rebalance.
    """
    if not use_internal_regime and not use_macro_regime:
        return cash_buffer, None

    internal_signal = None
    if use_internal_regime:
        try:
            internal_signal = regime.regime_signal(price_panel, index_close, as_of)
        except Exception as e:
            print(f"  [internal regime overlay unavailable: {e}]")

    macro_signal = None
    if use_macro_regime:
        try:
            macro_signal = macro.regime_signal(as_of)
        except Exception as e:
            print(f"  [macro regime overlay unavailable: {e}]")

    risk_off = (internal_signal or {}).get("risk_off", False) or (macro_signal or {}).get("risk_off", False)
    buffer = macro.RISK_OFF_CASH_BUFFER if risk_off else cash_buffer
    return buffer, {"internal": internal_signal, "macro": macro_signal, "risk_off": risk_off}


def build_plan(
    groww, top_n: int, max_weight: float, cash_buffer: float,
    use_internal_regime: bool = True, use_macro_regime: bool = False,
):
    symbols = get_universe()
    end = datetime.today()
    start = end - timedelta(days=LOOKBACK_CALENDAR_DAYS)

    print(f"Updating history for {len(symbols)} symbols...")
    history = fetch_universe_history(symbols, start, end)
    nifty_close = fetch_history("NIFTY", start, end)["close"]

    price_panel = factors.build_close_panel(history)
    turnover_panel = factors.build_turnover_panel(history)
    as_of = price_panel.index.max()
    print(f"Signals as of {as_of.date()}")

    cash_buffer, signals = resolve_cash_buffer(
        price_panel, nifty_close, as_of, cash_buffer, use_internal_regime, use_macro_regime
    )
    if signals is not None:
        state = "RISK-OFF" if signals["risk_off"] else "risk-on"
        parts = []
        if signals["internal"] is not None and signals["internal"]["breadth"] is not None:
            parts.append(
                f"breadth {signals['internal']['breadth']*100:.0f}%, "
                f"vol {signals['internal']['realized_vol']*100:.1f}% vs median "
                f"{signals['internal']['median_vol']*100:.1f}%"
            )
        if signals["macro"] is not None and signals["macro"]["usdinr_change_pct"] is not None:
            parts.append(
                f"USD/INR {signals['macro']['usdinr']:.2f} ({signals['macro']['usdinr_change_pct']*100:+.1f}%), "
                f"US 10Y {signals['macro']['yield_10y']:.2f}% ({signals['macro']['yield_10y_change_bps']:+.0f}bps)"
            )
        detail = "; ".join(parts)
        print(f"Regime: {state}  ({detail})  -> cash buffer {cash_buffer*100:.0f}%")

    target_weights = factors.select_portfolio(
        price_panel, turnover_panel, as_of, top_n=top_n, max_weight=max_weight
    )

    holdings_resp = groww.get_holdings_for_user()
    holdings = {h["trading_symbol"]: h["quantity"] for h in holdings_resp.get("holdings", [])}
    current_universe_holdings = {s: q for s, q in holdings.items() if s in price_panel.columns}

    margin = groww.get_available_margin_details()
    cash_available = margin["equity_margin_details"]["cnc_balance_available"]

    price_symbols = sorted(set(current_universe_holdings) | set(target_weights.index))
    live_prices = get_live_prices(groww, price_symbols)

    held_value = sum(
        current_universe_holdings.get(s, 0) * (live_prices.get(s) or 0) for s in price_symbols
    )
    total_capital = cash_available + held_value
    investable = total_capital * (1 - cash_buffer)

    plan = []
    for sym in price_symbols:
        price = live_prices.get(sym)
        if not price:
            continue
        current_qty = current_universe_holdings.get(sym, 0)
        target_value = investable * target_weights.get(sym, 0.0)
        target_qty = int(target_value / price)
        delta_qty = target_qty - current_qty
        trade_value = abs(delta_qty) * price
        if delta_qty == 0 or trade_value < MIN_ORDER_VALUE:
            continue
        plan.append({
            "symbol": sym,
            "side": "BUY" if delta_qty > 0 else "SELL",
            "quantity": abs(delta_qty),
            "price": price,
            "trade_value": trade_value,
            "current_qty": current_qty,
            "target_qty": target_qty,
            "target_weight": target_weights.get(sym, 0.0),
        })

    return {
        "as_of": as_of,
        "target_weights": target_weights,
        "total_capital": total_capital,
        "cash_available": cash_available,
        "investable": investable,
        "plan": plan,
    }


def print_plan(result: dict):
    print(f"\nTotal capital (cash + held): Rs {result['total_capital']:,.0f}")
    print(f"Cash available:              Rs {result['cash_available']:,.0f}")
    print(f"Investable (after buffer):   Rs {result['investable']:,.0f}")
    print(f"\nTarget portfolio ({len(result['target_weights'])} names):")
    for sym, w in result["target_weights"].sort_values(ascending=False).items():
        print(f"  {sym:<15} {w*100:5.2f}%")

    if not result["plan"]:
        print("\nNo trades needed -- current holdings already match target.")
        return

    print(f"\nPlanned orders ({len(result['plan'])}):")
    for o in result["plan"]:
        print(
            f"  {o['side']:<4} {o['symbol']:<15} qty={o['quantity']:<6} "
            f"@ ~Rs{o['price']:.2f}  (value Rs {o['trade_value']:,.0f})"
        )


def log_trade(row: dict):
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    write_header = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def execute_plan(groww, plan: list[dict]):
    for o in plan:
        response = groww.place_order(
            trading_symbol=o["symbol"],
            quantity=o["quantity"],
            validity=GrowwAPI.VALIDITY_DAY,
            exchange=GrowwAPI.EXCHANGE_NSE,
            segment=GrowwAPI.SEGMENT_CASH,
            product=GrowwAPI.PRODUCT_CNC,
            order_type=GrowwAPI.ORDER_TYPE_MARKET,
            transaction_type=(
                GrowwAPI.TRANSACTION_TYPE_BUY if o["side"] == "BUY" else GrowwAPI.TRANSACTION_TYPE_SELL
            ),
            price=0.0,
            trigger_price=None,
        )
        print(f"  -> {o['side']} {o['symbol']} x{o['quantity']}: {response}")
        log_trade({
            "timestamp": datetime.now().isoformat(),
            "symbol": o["symbol"],
            "side": o["side"],
            "quantity": o["quantity"],
            "ref_price": o["price"],
            "order_response": str(response),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=18)
    parser.add_argument("--max-weight", type=float, default=0.12)
    parser.add_argument("--cash-buffer", type=float, default=0.05)
    parser.add_argument(
        "--no-internal-regime",
        action="store_true",
        help="Disable the internal breadth+volatility regime overlay (on by default; validated).",
    )
    parser.add_argument(
        "--macro-regime",
        action="store_true",
        help="Enable the Alpha Vantage macro overlay (off by default; backtested worse than nothing).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually place the orders. Without this flag, just prints the plan.",
    )
    args = parser.parse_args()

    groww = get_client()
    result = build_plan(
        groww, args.top_n, args.max_weight, args.cash_buffer,
        use_internal_regime=not args.no_internal_regime, use_macro_regime=args.macro_regime,
    )
    print_plan(result)

    if not result["plan"]:
        return

    if not args.confirm:
        print("\nDRY RUN -- no orders sent. Re-run with --confirm to place them.")
        return

    print("\nPlacing orders...")
    execute_plan(groww, result["plan"])


if __name__ == "__main__":
    main()
