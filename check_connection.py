"""Verify Groww API credentials work by fetching the user profile and holdings."""
from client import get_client

if __name__ == "__main__":
    groww = get_client()

    profile = groww.get_user_profile()
    print("Connected as:", profile)

    margin = groww.get_available_margin_details()
    print("Available margin:", margin)

    holdings = groww.get_holdings_for_user()
    print("Holdings:", holdings)
