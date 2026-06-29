"""Create (or reuse) the LiveKit outbound SIP trunk that points at Plivo.

Reads the Plivo values from .env.local and registers an outbound trunk with
LiveKit Cloud, then prints the trunk id to copy into .env.local as
LIVEKIT_SIP_OUTBOUND_TRUNK_ID. Safe to re-run: if a trunk already exists for the
same Plivo termination domain, it is reused instead of creating a duplicate.

Run once:
    uv run python telephony/create_outbound_trunk.py
"""

import asyncio
import os

from dotenv import load_dotenv
from livekit import api

load_dotenv(".env.local")

TRUNK_NAME = "livekit-outbound (plivo)"


async def main() -> None:
    address = os.environ["TERMINATION_SIP"]  # e.g. 3033...zt.plivo.com
    number = os.environ["PLIVO_NUMBER"]  # caller id, must be a Plivo number
    username = os.environ["SIP_USERNAME"]
    password = os.environ["SIP_PASSWORD"]

    lk = api.LiveKitAPI()  # url/key/secret come from LIVEKIT_* env vars
    try:
        # Reuse an existing trunk with the same Plivo termination address so
        # re-running this script never piles up duplicate trunks.
        try:
            existing = await lk.sip.list_sip_outbound_trunk(
                api.ListSIPOutboundTrunkRequest()
            )
            for trunk in existing.items:
                if trunk.address == address:
                    print(f"Reusing existing trunk for {address}")
                    print(f"LIVEKIT_SIP_OUTBOUND_TRUNK_ID={trunk.sip_trunk_id}")
                    return
        except Exception as exc:  # listing is best-effort
            print(f"(could not list existing trunks: {exc}; creating a new one)")

        trunk = api.SIPOutboundTrunkInfo(
            name=TRUNK_NAME,
            address=address,
            transport=api.SIPTransport.SIP_TRANSPORT_TCP,  # Secure Trunking off
            numbers=[number],
            auth_username=username,
            auth_password=password,
        )
        created = await lk.sip.create_sip_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(trunk=trunk)
        )
        print("Created outbound trunk.")
        print(f"LIVEKIT_SIP_OUTBOUND_TRUNK_ID={created.sip_trunk_id}")
        print("\nAdd that line to .env.local, then run telephony/dial.py to test.")
    finally:
        await lk.aclose()


if __name__ == "__main__":
    asyncio.run(main())
