#!/usr/bin/env python3
"""Retrieval regression gate over the frozen Seattle snapshot.

Uses the same FastEmbed + FAISS retriever as the app. Does not import the
RentCast fetch script or scraper, and does not call Groq / OpenAI / RentCast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leasegpt.listings import SAMPLE_LISTINGS, SNAPSHOT_FETCHED_AT
from leasegpt.retrieval_eval import (
    DEFAULT_GOLD_PATH,
    DEFAULT_THRESHOLDS,
    EVAL_K,
    check_thresholds,
    evaluate_sources,
    format_report,
    load_gold,
)
from leasegpt.retriever import get_set_vector_store, get_text_chunks, retrieve_sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=DEFAULT_GOLD_PATH,
        help="Path to retrieval gold JSON (default: tests/eval/retrieval_gold.json)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=EVAL_K,
        help="Chunks to retrieve per query (default: 10 so hit@4 and hit@10 share a list)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable metrics after the text report",
    )
    args = parser.parse_args()

    gold_queries, payload = load_gold(args.gold)
    expected_date = str(payload.get("snapshot_fetched_at") or "").strip()
    if expected_date and expected_date != SNAPSHOT_FETCHED_AT:
        print(
            f"WARNING: gold snapshot_fetched_at={expected_date} "
            f"but listings report {SNAPSHOT_FETCHED_AT}",
            file=sys.stderr,
        )

    chunks = get_text_chunks("")
    vector_store = get_set_vector_store(chunks, "Seattle")

    def retrieve(query: str, k: int):
        return retrieve_sources(vector_store, query, k=k)

    report = evaluate_sources(
        gold_queries,
        retrieve,
        k=args.k,
        snapshot_fetched_at=SNAPSHOT_FETCHED_AT,
        listings=SAMPLE_LISTINGS,
    )
    failures = check_thresholds(report, DEFAULT_THRESHOLDS)
    sys.stdout.write(format_report(report, failures))
    if args.json:
        json.dump(report.as_dict(), sys.stdout, indent=2)
        sys.stdout.write("\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
