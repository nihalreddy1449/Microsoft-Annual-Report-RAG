"""Pre-download the models the pipeline needs, with timeouts and retries.

Downloading lazily on first use is what hung this project once already: the
Hugging Face CDN connection dropped into CloseWait mid-file, and because the
downloader had no socket timeout it waited indefinitely - twelve minutes of
wall clock for twelve seconds of CPU, with a half-written .incomplete blob and
no output to show why.

Fetching models explicitly, with HF_HUB_DOWNLOAD_TIMEOUT set and bounded
retries, turns that silent hang into a visible, recoverable failure. Partial
downloads resume, so a retry does not start over.

Run:  python scripts/fetch_models.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Must be set before huggingface_hub is imported, since it reads the value at
# import time. Without it a stalled socket blocks forever.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")

MODELS = [
    ("BAAI/bge-base-en-v1.5", "embeddings (step 5)"),
    ("BAAI/bge-reranker-base", "cross-encoder reranking (step 8)"),
]

MAX_ATTEMPTS = 5


def fetch(repo_id: str, purpose: str) -> bool:
    from huggingface_hub import snapshot_download

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            print(f"  attempt {attempt}/{MAX_ATTEMPTS} ...", flush=True)
            started = time.time()
            path = snapshot_download(
                repo_id=repo_id,
                # These repos ship the same weights several times over. Fetch
                # only safetensors, which transformers prefers anyway: leaving
                # pytorch_model.bin in doubled bge-base from 419MB to 739MB and
                # was the slow file still downloading long after the model was
                # already usable.
                ignore_patterns=[
                    "*.h5", "*.ot", "*.msgpack", "*onnx*", "*openvino*",
                    "pytorch_model.bin", "*.bin",
                ],
            )
            # Excluding the .bin weights is only safe if safetensors is
            # actually present, so confirm rather than assume.
            weights = list(Path(path).glob("*.safetensors"))
            if not weights:
                print("  no safetensors in repo - refetching with .bin allowed", flush=True)
                path = snapshot_download(
                    repo_id=repo_id,
                    ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*onnx*", "*openvino*"],
                )
            size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
            print(f"  OK ({time.time() - started:.0f}s, {size / 1024**2:.0f} MB) -> {path}", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001 - report and retry anything
            print(f"  failed: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            if attempt < MAX_ATTEMPTS:
                wait = min(2**attempt, 20)
                print(f"  retrying in {wait}s (partial download resumes)", flush=True)
                time.sleep(wait)
    return False


def main() -> int:
    print(f"HF_HUB_DOWNLOAD_TIMEOUT={os.environ['HF_HUB_DOWNLOAD_TIMEOUT']}s\n", flush=True)
    failed: list[str] = []
    for repo_id, purpose in MODELS:
        print(f"{repo_id}  ({purpose})", flush=True)
        if not fetch(repo_id, purpose):
            failed.append(repo_id)
        print(flush=True)

    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("Re-run this script; downloads resume from where they stopped.")
        return 1
    print("All models cached locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
