"""CLI: build the knowledge-base index.

    python -m insurex_agent.ingestion.build_index
"""

from __future__ import annotations

import argparse
import json
import logging

from ..config import get_settings


def main() -> None:
    ap = argparse.ArgumentParser(description="Parse the 5 product PDFs and build the hybrid index.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO, format="%(levelname)s %(message)s")
    from ..retrieval.store import build_index

    from ..llm import check_credentials

    s = get_settings()
    check_credentials(s, need_llm=False)
    print(f"Embedding provider: {s.embedding_provider} / {s.embedding_model}")
    report = build_index(s)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
