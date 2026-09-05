"""Verify the Groq API key works and report which models are actually served.

Model availability on free tiers shifts, so the generator and judge model ids
are read from .env rather than hardcoded, and this script lists what the
account can currently reach instead of trusting a name from memory.

Never prints the key.

Run:  python scripts/check_groq.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Models worth considering for either role, in rough preference order. Anything
# served but unlisted here still shows up in the output.
PREFERRED_GENERATORS = ["llama-3.3-70b", "llama-3.1-70b", "llama3-70b"]


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")

    key = os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        print("GROQ_API_KEY is empty in .env")
        print("Get one free at https://console.groq.com -> API Keys")
        return 1
    if not key.startswith("gsk_"):
        print(f"GROQ_API_KEY does not look like a Groq key (expected 'gsk_' prefix, "
              f"got {len(key)} chars starting {key[:4]!r})")
        return 1
    print(f"key present ({len(key)} chars, value not shown)\n")

    from groq import Groq

    client = Groq(api_key=key)

    try:
        models = sorted(m.id for m in client.models.list().data)
    except Exception as exc:  # noqa: BLE001
        print(f"could not list models: {type(exc).__name__}: {str(exc)[:200]}")
        print("If this is a 401, the key is invalid or was revoked - rotate it in the console.")
        return 1

    chat_models = [m for m in models if not any(x in m for x in ("whisper", "tts", "guard"))]
    print(f"{len(models)} models served, {len(chat_models)} usable for chat:\n")
    for model in chat_models:
        print(f"  {model}")

    configured = os.environ.get("GROQ_MODEL", "").strip()
    judge = os.environ.get("GROQ_JUDGE_MODEL", "").strip()
    print(f"\nGROQ_MODEL       = {configured or '(unset)'}")
    print(f"GROQ_JUDGE_MODEL = {judge or '(unset)'}")

    if configured and configured not in models:
        print(f"\n  WARNING: GROQ_MODEL {configured!r} is not in the served list above.")
    if judge and judge == configured:
        print("\n  WARNING: judge and generator are the same model. Locked decision 5 "
              "requires them to differ, to avoid self-preference bias.")

    # A real round trip, so a bad key or a dead model fails here and not
    # halfway through the eval harness.
    target = configured if configured in models else None
    if target is None:
        for prefix in PREFERRED_GENERATORS:
            match = next((m for m in chat_models if m.startswith(prefix)), None)
            if match:
                target = match
                break
    if target is None:
        target = chat_models[0] if chat_models else None
    if target is None:
        print("\nno chat model available to test")
        return 1

    print(f"\ntest completion against {target} ...")
    started = time.time()
    try:
        response = client.chat.completions.create(
            model=target,
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=8,
            temperature=0,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  failed: {type(exc).__name__}: {str(exc)[:200]}")
        return 1

    elapsed = time.time() - started
    reply = (response.choices[0].message.content or "").strip()
    usage = response.usage
    print(f"  reply: {reply!r}")
    print(f"  latency: {elapsed * 1000:.0f} ms")
    if usage:
        print(f"  tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
    print("\nGroq is reachable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
