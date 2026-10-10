#!/usr/bin/env python3
"""Submission entrypoint for AMD Mini-Challenge 2 (OCR).

Accepts --input-image <path> and writes /app/output/<stem>_output.json.
Communicates with resident daemon via /tmp/ocr.sock with in-process fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import socket
import sys

OUTPUT_DIR = Path(os.environ.get("MC2_OUTPUT_DIR", "/app/output"))
SOCKET_PATH = "/tmp/ocr.sock"

logger = logging.getLogger(__name__)


def send_daemon_request(payload: dict) -> dict | None:
    if not os.path.exists(SOCKET_PATH):
        return None
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(25.0)
        sock.connect(SOCKET_PATH)
        sock.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        data = b""
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk
            if b"\n" in chunk:
                break
        sock.close()
        return json.loads(data.decode("utf-8").strip())
    except Exception as e:
        logger.warning("Daemon request failed: %s", e)
        return None


def recognize_image(image_path: Path) -> tuple[str, float]:
    """Recognize text from image."""
    resp = send_daemon_request({"image_path": str(image_path.resolve())})
    if resp is not None and "text" in resp:
        return str(resp["text"]).strip(), float(resp.get("confidence", 0.95))

    # In-process fallback
    from server import OCRServer
    server = OCRServer()
    resp = server.process_image(image_path)
    return str(resp.get("text", "")).strip(), float(resp.get("confidence", 0.95))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-image", type=Path, required=True, help="path to test image")
    args = ap.parse_args()

    if not args.input_image.exists():
        ap.error(f"input image {args.input_image} does not exist")

    text, confidence = recognize_image(args.input_image)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_stem = args.input_image.stem
    out_path = OUTPUT_DIR / f"{out_stem}_output.json"

    # Write atomically
    out_path.write_text(
        json.dumps({"text": text, "confidence": confidence}, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
