"""Place an order on Groww.

Places a real order that can execute with real money. Defaults to a dry run
that only prints what would be sent — pass --confirm to actually place it.

Example:
    python place_order.py --symbol WIPRO --qty 1 --side BUY --type LIMIT --price 250 --confirm
"""
import argparse

from growwapi import GrowwAPI

from client import get_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. WIPRO")
    parser.add_argument("--qty", type=int, required=True)
    parser.add_argument("--side", choices=["BUY", "SELL"], required=True)
    parser.add_argument("--type", choices=["MARKET", "LIMIT"], default="MARKET")
    parser.add_argument("--price", type=float, default=0.0, help="Required for LIMIT orders")
    parser.add_argument("--trigger-price", type=float, default=None)
    parser.add_argument("--product", choices=["CNC", "MIS", "NRML"], default="CNC")
    parser.add_argument("--exchange", choices=["NSE", "BSE"], default="NSE")
    parser.add_argument("--reference-id", default=None)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually send the order. Without this flag, just prints the request.",
    )
    args = parser.parse_args()

    order_kwargs = dict(
        trading_symbol=args.symbol,
        quantity=args.qty,
        validity=GrowwAPI.VALIDITY_DAY,
        exchange=getattr(GrowwAPI, f"EXCHANGE_{args.exchange}"),
        segment=GrowwAPI.SEGMENT_CASH,
        product=getattr(GrowwAPI, f"PRODUCT_{args.product}"),
        order_type=getattr(GrowwAPI, f"ORDER_TYPE_{args.type}"),
        transaction_type=getattr(GrowwAPI, f"TRANSACTION_TYPE_{args.side}"),
        price=args.price,
        trigger_price=args.trigger_price,
        order_reference_id=args.reference_id,
    )

    if not args.confirm:
        print("DRY RUN — no order sent. Re-run with --confirm to place it.")
        print("Would call groww.place_order(**", order_kwargs, ")")
        return

    groww = get_client()
    response = groww.place_order(**order_kwargs)
    print("Order placed:", response)


if __name__ == "__main__":
    main()
