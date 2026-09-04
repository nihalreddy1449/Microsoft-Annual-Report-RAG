"""Verify the environment is actually usable before building on top of it.

The important check here is NOT `torch.cuda.is_available()`. On a Blackwell
GPU (RTX 50-series, compute capability sm_120) a torch build compiled without
sm_120 kernels will import fine, report CUDA available, report the correct
device name -- and then fail on the first real kernel launch with
"no kernel image is available for execution on the device".

So this script runs an actual matmul on the GPU and reads the result back.
That is the only check that proves the toolchain works end to end.

Run:  python scripts/check_env.py
"""

from __future__ import annotations

import importlib
import platform
import sys
from importlib import metadata

OK, FAIL, WARN = "[ OK ]", "[FAIL]", "[WARN]"


def check_torch() -> bool:
    try:
        import torch
    except ImportError:
        print(f"{FAIL} torch not installed.")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return False

    print(f"{OK} torch {torch.__version__} (built for CUDA {torch.version.cuda})")

    if not torch.cuda.is_available():
        print(f"{WARN} No CUDA device visible. The project still runs on CPU, just slower.")
        return True

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"{OK} GPU: {name}  (sm_{major}{minor}, {vram:.1f} GB VRAM)")

    supported = torch.cuda.get_arch_list()
    print(f"       torch was compiled for: {', '.join(supported)}")

    arch = f"sm_{major}{minor}"
    if arch not in supported:
        print(f"{WARN} {arch} is NOT in this build's arch list - a kernel launch will likely fail.")

    # The real test: force an actual kernel launch and read the result back.
    try:
        a = torch.randn(512, 512, device="cuda")
        b = torch.randn(512, 512, device="cuda")
        c = (a @ b).sum().item()
        torch.cuda.synchronize()
        print(f"{OK} GPU matmul executed and returned a value ({c:.2f}) - kernels work.")
    except RuntimeError as exc:
        print(f"{FAIL} GPU kernel launch failed: {exc}")
        print("       Reinstall torch from the cu128 index (see requirements.txt).")
        return False

    return True


def check_imports() -> bool:
    """Import each dependency we rely on, reporting its installed version.

    Import name and distribution name differ for several of these (dotenv ->
    python-dotenv), so the version comes from package metadata rather than a
    __version__ attribute. Some packages (unstructured) expose __version__ as
    a submodule rather than a string, which is why the attribute is a fallback
    only and is always coerced to str.
    """
    # (import name, distribution name, purpose)
    # No parsing library: docx parsing uses the standard library, see
    # src/ingestion/docx_parser.py.
    packages = [
        ("sentence_transformers", "sentence-transformers", "bge embeddings + reranker"),
        ("transformers", "transformers", "model backend"),
        ("chromadb", "chromadb", "vector store"),
        ("rank_bm25", "rank-bm25", "sparse retrieval"),
        ("groq", "groq", "generation API"),
        ("dotenv", "python-dotenv", "loads .env"),
        ("numpy", "numpy", "numerics"),
        ("pandas", "pandas", "results tables"),
        ("gradio", "gradio", "interface"),
    ]
    all_ok = True
    for module, dist, purpose in packages:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            print(f"{FAIL} {module:<24} {'--':<12} {purpose} - not installed")
            all_ok = False
            continue

        try:
            version = metadata.version(dist)
        except metadata.PackageNotFoundError:
            version = str(getattr(mod, "__version__", "?"))
        print(f"{OK} {module:<24} {version:<12} {purpose}")
    return all_ok


def check_secrets() -> None:
    """Confirm .env exists and holds a key, without ever printing the value."""
    from pathlib import Path

    env = Path(__file__).resolve().parent.parent / ".env"
    if not env.exists():
        print(f"{WARN} .env not found. Copy .env.example to .env and add your Groq key.")
        return

    key = ""
    for line in env.read_text(encoding="utf-8").splitlines():
        if line.startswith("GROQ_API_KEY="):
            key = line.split("=", 1)[1].strip()

    if not key:
        print(f"{WARN} .env exists but GROQ_API_KEY is empty (only needed from step 9 onward).")
    elif not key.startswith("gsk_"):
        print(f"{WARN} GROQ_API_KEY does not look like a Groq key (expected 'gsk_' prefix).")
    else:
        print(f"{OK} GROQ_API_KEY present ({len(key)} chars, value not shown).")


def main() -> int:
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    print(f"Interpreter: {sys.executable}\n")

    print("--- torch / GPU ---")
    torch_ok = check_torch()

    print("\n--- dependencies ---")
    imports_ok = check_imports()

    print("\n--- secrets ---")
    check_secrets()

    print()
    if torch_ok and imports_ok:
        print("Environment ready.")
        return 0
    print("Environment incomplete - see FAIL lines above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
