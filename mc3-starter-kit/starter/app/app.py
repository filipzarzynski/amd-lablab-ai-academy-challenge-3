#!/usr/bin/env python3
"""Submission entrypoint for AMD Mini-Challenge 3 (RAG).

Acts as a thin client to the resident background daemon via /tmp/rag.sock.
Falls back to direct in-process execution if running outside daemon mode.
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
SOCKET_PATH = "/tmp/rag.sock"

logger = logging.getLogger(__name__)


def send_daemon_request(payload: dict) -> dict | None:
    """Send request to the resident daemon over Unix socket."""
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


def index(corpus: Path) -> None:
    """Build and persist whatever is needed to answer questions."""
    resp = send_daemon_request({"cmd": "index", "corpus": str(corpus.resolve())})
    if resp is not None:
        return

    # Fallback to direct in-process server if daemon is not running
    from server import RAGServer
    server = RAGServer()
    server.index_corpus(corpus)


def answer(corpus: Path, query: str) -> tuple[str, list[str], float]:
    """Return (answer, citations, confidence) for ONE question."""
    resp = send_daemon_request({
        "cmd": "query",
        "corpus": str(corpus.resolve()),
        "query": query,
    })
    if resp is not None and "answer" in resp and "citations" in resp:
        return resp["answer"], resp["citations"], float(resp.get("confidence", 0.9))

    # Fallback to direct in-process server
    from server import RAGServer
    server = RAGServer()
    resp = server.answer_query(query, corpus)
    return resp["answer"], resp["citations"], float(resp.get("confidence", 0.9))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, help="build the index over this corpus, then exit")
    ap.add_argument("--corpus", type=Path, help="corpus root for a query")
    ap.add_argument("--query-id", help="output stem the harness assigns, e.g. query_01")
    ap.add_argument("--query", help="the question to answer")
    args = ap.parse_args()

    if args.index is not None:
        index(args.index)
        return 0

    if args.corpus is None or args.query is None or not args.query_id:
        ap.error("a query needs --corpus, --query-id and --query")

    text, citations, confidence = answer(args.corpus, args.query)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / (args.query_id + "_output.json")
    # Written whole to avoid partial-read parse errors by the harness
    out.write_text(
        json.dumps(
            {"answer": text, "citations": list(citations), "confidence": confidence},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
