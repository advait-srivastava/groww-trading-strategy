"""Daily circuit breaker: sell any position that has breached its stop-loss.

Runs independently of the monthly rebalance (strategy/rebalance.py) so a
sharp drawdown between rebalances doesn't sit unmanaged for weeks. Only ever
sells -- never buys or resizes -- so it's safe to run daily via cron/launchd
alongside the monthly rebalance.

Example:
    python -m strategy.risk_check              # dry run
    python -m strategy.risk_check --confirm     # sells breached positions
"""
import argparse

from growwapi import GrowwAPI

from client import get_client
from strategy.rebalance import execute_plan, get_live_prices

STOP_LOSS_PCT = 0.18  # exit if a position is down >18% from its average buy price


def find_breaches(groww, stop_loss_pct: float) -> list[dict]:
    holdings = groww.get_holdings_for_user().get("holdings", [])
    symbols = [h["trading_symbol"] for h in holdings if h["quantity"] > 0]
    prices = get_live_prices(groww, symbols)

    breaches = []
    for h in holdings:
        sym, qty, avg_price = h["trading_symbol"], h["quantity"], h["average_price"]
        if qty <= 0 or not avg_price:
            continue
        ltp = prices.get(sym)
        if not ltp:
            continue
        loss_pct = (ltp - avg_price) / avg_price
        if loss_pct <= -stop_loss_pct:
            breaches.append({
                "symbol": sym,
                "side": "SELL",
                "quantity": int(qty),
                "price": ltp,
                "trade_value": qty * ltp,
                "avg_price": avg_price,
                "loss_pct": loss_pct,
            })
    return breaches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-loss-pct", type=float, default=STOP_LOSS_PCT)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()

    groww = get_client()
    breaches = find_breaches(groww, args.stop_loss_pct)

    if not breaches:
        print("No stop-loss breaches -- all positions within tolerance.")
        return

    print(f"Stop-loss breached on {len(breaches)} position(s):")
    for b in breaches:
        print(
            f"  {b['symbol']:<15} avg Rs{b['avg_price']:.2f} -> "
            f"now Rs{b['price']:.2f}  ({b['loss_pct']*100:.1f}%)"
        )

    if not args.confirm:
        print("\nDRY RUN -- no orders sent. Re-run with --confirm to sell these positions.")
        return

    print("\nSelling breached positions...")
    execute_plan(groww, breaches)


if __name__ == "__main__":
    main()
