# Outbound calling via Plivo (Zentrunk) → LiveKit

This wires the voice agent to place **outbound** PSTN calls through a Plivo
Zentrunk outbound trunk. Inbound is not set up here.

## One-time values (in `.env.local`)

```
PLIVO_NUMBER=+912269986322          # rented Plivo number = caller id
TERMINATION_SIP=...zt.plivo.com     # Plivo Outbound Trunk "Termination SIP Domain"
SIP_USERNAME=...                    # Plivo trunk credential username
SIP_PASSWORD=...                    # Plivo trunk credential password  (secret)
LIVEKIT_SIP_OUTBOUND_TRUNK_ID=ST_...# created by create_outbound_trunk.py
```

Transport is **TCP** (Secure Trunking is off on the Plivo trunk). To switch to
TLS later, enable Secure Trunking in Plivo and change the transport in
`create_outbound_trunk.py` to `SIP_TRANSPORT_TLS`, then re-create the trunk.

## Setup (already done once)

```bash
uv run python telephony/create_outbound_trunk.py
# -> prints LIVEKIT_SIP_OUTBOUND_TRUNK_ID=ST_...  (already added to .env.local)
```

## Place a test call

1. **Destination must be a verified/sandbox number** on a Plivo trial — add your
   mobile at `cx.plivo.com` (Sandbox Numbers) and put the **same** number in the
   `Phone` cell of the loaded customer (Amit Verma's row) in
   `data/Customers.xlsx`, in E.164 (e.g. `+9198XXXXXXXX`).

2. **Start the agent** (registers as `my-agent`):
   ```bash
   uv run python src/agent.py dev
   ```

3. **Dial** (separate terminal):
   ```bash
   uv run python telephony/dial.py                 # dials the Excel Phone cell
   uv run python telephony/dial.py --to +9198XXXXXXXX   # or override
   ```
   Your phone rings; answer and talk to Priya.

## Notes / troubleshooting

- The dialer **dispatches `my-agent`** into a fresh room, then dials the customer
  into that room (named agents only join rooms they're dispatched to).
- For now the agent still loads **one** customer (Amit Verma) regardless of who
  is dialed — fine for a first end-to-end test. Per-customer "load by phone" is
  the next step.
- Call failures (e.g. trial trying to reach an unverified number) surface as an
  exception from `dial.py` and in Plivo's **Zentrunk logs** (`cx.plivo.com`) and
  the LiveKit call logs.
- Rotate `SIP_PASSWORD` in Plivo after testing; `.env.local` is gitignored but
  the credential is still a secret.
