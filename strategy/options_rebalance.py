"""Weekly NIFTY bull put credit spread: entry planning + exit monitoring.

Uses REAL option chain data (live deltas, LTPs, margin) for the actual
decision -- more accurate than strategy/options_backtest.py's Black-Scholes
approximation, which only exists because Groww has no historical option
data to validate against. Defaults to a dry run; pass --confirm to place
real orders. Same discipline as strategy/rebalance.py.

Entry:
    python -m strategy.options_rebalance entry              # dry run
    python -m strategy.options_rebalance entry --confirm    # places real orders

Exit check (run this daily against an open position):
    python -m strategy.options_rebalance exit                # dry run
    python -m strategy.options_rebalance exit --confirm      # closes if target hit
"""
import argparse
import csv
import json
import os
from datetime import datetime, timedelta

from growwapi import GrowwAPI

from client import get_client
from strategy import factors, options_strategy as ostrat, regime
from strategy.data import fetch_history, fetch_universe_history
from strategy.universe import get_universe

POSITION_FILE = os.path.join(os.path.dirname(__file__), "data_cache", "options_position.json")
TRADE_LOG = os.path.join(os.path.dirname(__file__), "data_cache", "options_trade_log.csv")
MIN_DAYS_TO_EXPIRY = 3  # skip expiries closer than this to avoid entering right into gamma risk


def get_regime(groww):
    symbols = get_universe()
    end = datetime.today()
    start = end - timedelta(days=800)
    history = fetch_universe_history(symbols, start, end)
    price_panel = factors.build_close_panel(history)
    nifty_close = fetch_history("NIFTY", start, end)["close"]
    as_of = price_panel.index.max()
    return regime.regime_signal(price_panel, nifty_close, as_of), as_of


def pick_expiry(groww) -> str:
    expiries = groww.get_expiries(exchange=GrowwAPI.EXCHANGE_NSE, underlying_symbol="NIFTY")["expiries"]
    today = datetime.today().date()
    for e in expiries:
        edate = datetime.strptime(e, "%Y-%m-%d").date()
        if (edate - today).days >= MIN_DAYS_TO_EXPIRY:
            return e
    return expiries[0]


def find_short_put(chain: dict, target_delta: float) -> tuple[str, dict]:
    best_strike, best_data, best_diff = None, None, float("inf")
    for strike_str, data in chain["strikes"].items():
        pe = data.get("PE")
        if not pe or pe["greeks"]["delta"] is None:
            continue
        diff = abs(pe["greeks"]["delta"] - target_delta)
        if diff < best_diff:
            best_strike, best_data, best_diff = strike_str, pe, diff
    return best_strike, best_data


def plan_entry(groww) -> dict:
    signal, as_of = get_regime(groww)
    result = {"as_of": as_of, "signal": signal, "trade": None, "skip_reason": None}

    if not ostrat.should_trade(signal):
        result["skip_reason"] = "regime is risk-off"
        return result

    expiry = pick_expiry(groww)
    chain = groww.get_option_chain(exchange=GrowwAPI.EXCHANGE_NSE, underlying="NIFTY", expiry_date=expiry)
    spot = chain["underlying_ltp"]

    short_strike_str, short_pe = find_short_put(chain, ostrat.SHORT_PUT_DELTA_TARGET)
    if short_strike_str is None:
        result["skip_reason"] = "no valid short strike found in chain"
        return result

    short_strike = float(short_strike_str)
    long_strike = short_strike - ostrat.SPREAD_WIDTH_POINTS
    long_strike_str = str(int(long_strike)) if long_strike == int(long_strike) else str(long_strike)
    long_data = chain["strikes"].get(long_strike_str, {}).get("PE")
    if long_data is None:
        result["skip_reason"] = f"no listed strike at {long_strike_str} for the protective leg"
        return result

    inst = groww.get_instrument_by_exchange_and_trading_symbol(
        exchange=GrowwAPI.EXCHANGE_NSE, trading_symbol=short_pe["trading_symbol"]
    )
    lot_size = int(inst["lot_size"])

    credit_per_unit = short_pe["ltp"] - long_data["ltp"]
    max_loss_per_unit = (short_strike - long_strike) - credit_per_unit

    margin = groww.get_order_margin_details(
        segment=GrowwAPI.SEGMENT_FNO,
        orders=[
            {"trading_symbol": short_pe["trading_symbol"], "transaction_type": "SELL", "quantity": lot_size,
             "order_type": "MARKET", "product": "NRML", "exchange": "NSE", "price": 0},
            {"trading_symbol": long_data["trading_symbol"], "transaction_type": "BUY", "quantity": lot_size,
             "order_type": "MARKET", "product": "NRML", "exchange": "NSE", "price": 0},
        ],
    )
    margin_required = margin["total_requirement"]

    if margin_required > ostrat.OPTIONS_BUDGET:
        result["skip_reason"] = (
            f"margin required (Rs {margin_required:,.0f}) exceeds the Rs {ostrat.OPTIONS_BUDGET:,.0f} options budget"
        )
        return result

    result["trade"] = {
        "expiry": expiry,
        "spot": spot,
        "short_strike": short_strike,
        "short_symbol": short_pe["trading_symbol"],
        "short_delta": short_pe["greeks"]["delta"],
        "short_ltp": short_pe["ltp"],
        "long_strike": long_strike,
        "long_symbol": long_data["trading_symbol"],
        "long_ltp": long_data["ltp"],
        "lot_size": lot_size,
        "credit_per_unit": credit_per_unit,
        "credit_total": credit_per_unit * lot_size,
        "max_loss_per_unit": max_loss_per_unit,
        "max_loss_total": max_loss_per_unit * lot_size,
        "margin_required": margin_required,
    }
    return result


def print_entry_plan(result: dict):
    sig = result["signal"]
    state = "RISK-OFF" if sig["risk_off"] else "risk-on"
    breadth = f"{sig['breadth']*100:.0f}%" if sig["breadth"] is not None else "n/a"
    vol = f"{sig['realized_vol']*100:.1f}%" if sig["realized_vol"] is not None else "n/a"
    print(f"Regime: {state}  (breadth {breadth}, realized vol {vol})")

    if result["skip_reason"]:
        print(f"\nNo trade: {result['skip_reason']}")
        return

    t = result["trade"]
    print(f"\nNIFTY spot: {t['spot']}")
    print(f"Expiry:     {t['expiry']}")
    print(f"\nBull put spread:")
    print(f"  SELL {t['short_symbol']:<20} (delta {t['short_delta']:.2f})  LTP {t['short_ltp']}")
    print(f"  BUY  {t['long_symbol']:<20}                    LTP {t['long_ltp']}")
    print(f"\n  Credit received:   Rs {t['credit_total']:,.0f} ({t['lot_size']} qty)")
    print(f"  Max loss:          Rs {t['max_loss_total']:,.0f}")
    print(f"  Margin required:   Rs {t['margin_required']:,.0f}  (budget: Rs {ostrat.OPTIONS_BUDGET:,.0f})")
    print(f"  Profit target:     close at Rs {t['credit_total']*ostrat.PROFIT_TARGET_PCT:,.0f} profit "
          f"({ostrat.PROFIT_TARGET_PCT*100:.0f}% of credit)")


def save_position(trade: dict):
    os.makedirs(os.path.dirname(POSITION_FILE), exist_ok=True)
    with open(POSITION_FILE, "w") as f:
        json.dump({**trade, "entered_at": datetime.now().isoformat()}, f, default=str, indent=2)


def load_position() -> dict | None:
    if not os.path.exists(POSITION_FILE):
        return None
    with open(POSITION_FILE) as f:
        return json.load(f)


def log_trade(row: dict):
    os.makedirs(os.path.dirname(TRADE_LOG), exist_ok=True)
    write_header = not os.path.exists(TRADE_LOG)
    with open(TRADE_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def place_spread(groww, trade: dict, side: str):
    """side='open': sell short/buy long. side='close': buy back short/sell long."""
    if side == "open":
        legs = [(trade["short_symbol"], "SELL"), (trade["long_symbol"], "BUY")]
    else:
        legs = [(trade["short_symbol"], "BUY"), (trade["long_symbol"], "SELL")]

    responses = []
    for symbol, txn in legs:
        resp = groww.place_order(
            trading_symbol=symbol,
            quantity=trade["lot_size"],
            validity=GrowwAPI.VALIDITY_DAY,
            exchange=GrowwAPI.EXCHANGE_NSE,
            segment=GrowwAPI.SEGMENT_FNO,
            product=GrowwAPI.PRODUCT_NRML,
            order_type=GrowwAPI.ORDER_TYPE_MARKET,
            transaction_type=GrowwAPI.TRANSACTION_TYPE_SELL if txn == "SELL" else GrowwAPI.TRANSACTION_TYPE_BUY,
            price=0.0,
            trigger_price=None,
        )
        print(f"  -> {txn} {symbol}: {resp}")
        responses.append(resp)
        log_trade({
            "timestamp": datetime.now().isoformat(), "side": side, "symbol": symbol,
            "transaction": txn, "quantity": trade["lot_size"], "order_response": str(resp),
        })
    return responses


def check_exit(groww) -> dict:
    position = load_position()
    if position is None:
        return {"has_position": False}

    chain = groww.get_option_chain(
        exchange=GrowwAPI.EXCHANGE_NSE, underlying="NIFTY", expiry_date=position["expiry"]
    )
    short_strike_str = str(int(position["short_strike"]))
    long_strike_str = str(int(position["long_strike"]))
    short_now = chain["strikes"].get(short_strike_str, {}).get("PE", {}).get("ltp")
    long_now = chain["strikes"].get(long_strike_str, {}).get("PE", {}).get("ltp")

    if short_now is None or long_now is None:
        return {"has_position": True, "position": position, "error": "strikes no longer in chain (expired?)"}

    current_spread_value = short_now - long_now
    credit_received = position["credit_per_unit"]
    profit_so_far = credit_received - current_spread_value
    target_profit = credit_received * ostrat.PROFIT_TARGET_PCT
    stop_level = credit_received * ostrat.STOP_LOSS_MULTIPLE

    should_close = profit_so_far >= target_profit or current_spread_value >= stop_level
    reason = None
    if profit_so_far >= target_profit:
        reason = "profit target hit"
    elif current_spread_value >= stop_level:
        reason = "stop level hit"

    return {
        "has_position": True, "position": position, "current_spread_value": current_spread_value,
        "profit_so_far_per_unit": profit_so_far, "profit_so_far_total": profit_so_far * position["lot_size"],
        "should_close": should_close, "reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["entry", "exit"])
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    groww = get_client()

    if args.action == "entry":
        if load_position() is not None:
            print("A position is already open (strategy/data_cache/options_position.json). "
                  "Run 'exit' to check/close it before entering a new one.")
            return

        result = plan_entry(groww)
        print_entry_plan(result)
        if result["trade"] is None:
            return
        if not args.confirm:
            print("\nDRY RUN -- no orders sent. Re-run with --confirm to place them.")
            return
        print("\nPlacing spread...")
        place_spread(groww, result["trade"], "open")
        save_position(result["trade"])

    else:
        status = check_exit(groww)
        if not status["has_position"]:
            print("No open position.")
            return
        if status.get("error"):
            print(f"Error checking position: {status['error']}")
            return

        p = status["position"]
        print(f"Position: SELL {p['short_symbol']} / BUY {p['long_symbol']}  (entered {p['entered_at']})")
        print(f"  Credit received:      Rs {p['credit_total']:,.0f}")
        print(f"  Current spread value: Rs {status['current_spread_value']*p['lot_size']:,.0f}")
        print(f"  P&L so far:           Rs {status['profit_so_far_total']:,.0f}")

        if not status["should_close"]:
            print("\nNo exit signal yet -- holding.")
            return

        print(f"\nExit signal: {status['reason']}")
        if not args.confirm:
            print("DRY RUN -- position not closed. Re-run with --confirm to close it.")
            return
        print("Closing position...")
        place_spread(groww, p, "close")
        os.remove(POSITION_FILE)


if __name__ == "__main__":
    main()
