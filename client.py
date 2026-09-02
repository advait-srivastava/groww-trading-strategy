"""Groww API client setup — handles both supported auth flows."""
import os

from dotenv import load_dotenv
from growwapi import GrowwAPI

load_dotenv()


def get_client() -> GrowwAPI:
    api_key = os.environ["GROWW_API_KEY"]
    secret = os.environ.get("GROWW_API_SECRET")
    totp_secret = os.environ.get("GROWW_TOTP_SECRET")

    if totp_secret:
        import pyotp

        totp = pyotp.TOTP(totp_secret).now()
        access_token = GrowwAPI.get_access_token(api_key=api_key, totp=totp)
    elif secret:
        access_token = GrowwAPI.get_access_token(api_key=api_key, secret=secret)
    else:
        raise RuntimeError(
            "Set GROWW_API_SECRET or GROWW_TOTP_SECRET in .env (see .env.example)"
        )

    return GrowwAPI(access_token)
