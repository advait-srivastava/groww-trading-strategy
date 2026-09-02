"""Fetch holdings and positions from Groww."""
from client import get_client

if __name__ == "__main__":
    groww = get_client()

    print("=== Holdings ===")
    print(groww.get_holdings_for_user())

    print("\n=== Positions ===")
    print(groww.get_positions_for_user())
