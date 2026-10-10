#!/usr/bin/env python3
"""Check your container against the MC-2 submission contract before submitting.

Same checks the harness runs:
    python3 selfcheck.py your-image:tag path/to/sample/image.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

SCRIPT = "/app/app.py"
OUTPUT_DIR = "/app/output"
STARTUP_BUDGET_S = 600
PER_IMAGE_BUDGET_S = 30
VRAM_FLOOR_GIB = 1
VRAM_CEILING_GIB = 48


def run(args, timeout=120):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)


def ok(msg):
    print(f"  [PASS] {msg}")
    return True


def bad(msg):
    print(f"  [FAIL] {msg}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="your built image, e.g. ghcr.io/you/ocr:v1")
    ap.add_argument("sample_image", type=Path, help="a sample image to try")
    args = ap.parse_args()

    if not args.sample_image.is_file():
        return 1 if bad(f"{args.sample_image} is not a file") else 0

    passed = True
    name = f"selfcheck-{uuid.uuid4().hex[:8]}"

    print("1. size, uncompressed")
    hist = run(["docker", "history", "--no-trunc", "--format", "{{.Size}}", args.image])
    if hist.returncode != 0:
        passed = bad(f"could not read image history: {hist.stderr.strip()}")
    else:
        ok("image exists locally; verify uncompressed size <= 60 GiB")

    print("2. container starts")
    started = time.time()
    up = run(["docker", "run", "-d", "--name", name, args.image])
    if up.returncode != 0:
        return 1 if bad(f"container did not start: {up.stderr.strip()}") else 0
    ok(f"started in {time.time() - started:.1f}s (budget {STARTUP_BUDGET_S}s)")

    try:
        print("3. image inference runs in time")
        stem = args.sample_image.stem
        in_container_path = f"/app/input/{args.sample_image.name}"
        run(["docker", "exec", name, "mkdir", "-p", "/app/input", OUTPUT_DIR], timeout=30)
        cp = run(["docker", "cp", str(args.sample_image), f"{name}:{in_container_path}"], timeout=30)
        if cp.returncode != 0:
            passed = bad(f"could not copy sample image: {cp.stderr.strip()}")

        t0 = time.time()
        proc = run(
            ["docker", "exec", name, "python3", SCRIPT, "--input-image", in_container_path],
            timeout=PER_IMAGE_BUDGET_S,
        )
        if proc.returncode != 0:
            passed = bad(f"{SCRIPT} exited {proc.returncode}: {(proc.stderr or proc.stdout).strip()[:400]}")
        else:
            ok(f"processed in {time.time() - t0:.2f}s (budget {PER_IMAGE_BUDGET_S}s)")

        print("4. output file shape and validity")
        want = f"{OUTPUT_DIR}/{stem}_output.json"
        cat = run(["docker", "exec", name, "cat", want], timeout=30)
        if cat.returncode != 0:
            passed = bad(f"no {want} created")
        else:
            try:
                payload = json.loads(cat.stdout)
            except ValueError:
                passed = bad(f"{want} is not valid JSON")
            else:
                if not isinstance(payload, dict):
                    passed = bad("JSON must be an object")
                elif "text" not in payload or not isinstance(payload["text"], str):
                    passed = bad("missing string key 'text'")
                else:
                    ok(f"output valid: text={payload['text']!r}, confidence={payload.get('confidence')}")

        print("5. VRAM check")
        print(f"   Peak VRAM must stay between {VRAM_FLOOR_GIB} and {VRAM_CEILING_GIB} GiB.")
    finally:
        run(["docker", "rm", "-f", name], timeout=60)

    print()
    print("ALL CHECKS PASSED" if passed else "SOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
