"""Throwaway connectivity check for Claude Haiku 4.5 on Bedrock (Japan region).

Validates the AWS side in isolation — billing, IAM (bedrock:InvokeModel), and the
Japan-only "jp." cross-region inference profile — before running the full voice
agent. Reads AWS creds from the environment (loaded from .env.local).

Run:  uv run python scripts/bedrock_jp_ping.py

Delete this file once the Japan Bedrock test is done.
"""

from __future__ import annotations

import os
import sys
import time

# Windows consoles default to cp1252, which can't print the Hindi (Devanagari) reply.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

REGION = os.environ.get("AWS_REGION", "ap-south-1")
MODEL_ID = os.environ.get("BEDROCK_MODEL", "mistral.voxtral-mini-3b-2507")


def main() -> int:
    # Load .env.local the same way the agent does, so creds/region match.
    load_dotenv(".env.local")

    # Creds are namespaced BEDROCK_AWS_* so they don't flip agent.py onto Bedrock;
    # promote them to the standard AWS_* names boto3 expects (matches
    # agent_v1_new.py's _build_agent_llm).
    if os.environ.get("BEDROCK_AWS_ACCESS_KEY_ID"):
        os.environ.setdefault(
            "AWS_ACCESS_KEY_ID", os.environ["BEDROCK_AWS_ACCESS_KEY_ID"]
        )
        os.environ.setdefault(
            "AWS_SECRET_ACCESS_KEY", os.environ["BEDROCK_AWS_SECRET_ACCESS_KEY"]
        )

    if not os.environ.get("AWS_ACCESS_KEY_ID"):
        print("BEDROCK_AWS_ACCESS_KEY_ID not set — set it in .env.local first.")
        return 1

    print(f"Region:   {REGION}")
    print(f"Model:    {MODEL_ID}")
    print("Sending a one-line prompt via Bedrock Converse...\n")

    client = boto3.client("bedrock-runtime", region_name=REGION)
    started = time.perf_counter()
    try:
        resp = client.converse(
            modelId=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"text": "Reply in one short Hindi sentence: are you working?"}
                    ],
                }
            ],
            inferenceConfig={"maxTokens": 64, "temperature": 0},
        )
    except ClientError as e:
        elapsed = (time.perf_counter() - started) * 1000
        code = e.response.get("Error", {}).get("Code", "Unknown")
        print(f"FAILED after {elapsed:.0f} ms — {code}: {e}")
        print(
            "\nHints: AccessDenied -> IAM/region or first-time Anthropic use-case gate; "
            "payment error -> billing; ValidationException -> wrong profile id for region."
        )
        return 1

    elapsed = (time.perf_counter() - started) * 1000
    reply = resp["output"]["message"]["content"][0]["text"]
    usage = resp.get("usage", {})
    print(f"OK in {elapsed:.0f} ms")
    print(f"Reply:    {reply}")
    print(f"Tokens:   {usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
