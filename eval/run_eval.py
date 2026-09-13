"""Run LeaseGPT retrieval and generation evaluation end to end."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import mean
import time
from typing import Any

from dotenv import load_dotenv

from eval.catalog import load_catalog, validate_relevant_ids
from eval.faithfulness import JUDGE_RUBRIC, evaluate_generation
from eval.html_report import render_html
from eval.retrieval_metrics import DEFAULT_KS, metrics_for_query, summarize
from eval.retrievers import CONFIGS, build_eval_store, retrieve
from leasegpt.listings import SNAPSHOT_DIGEST, SNAPSHOT_FETCHED_AT
from leasegpt.retriever import FASTEMBED_MODEL


EVAL_DIR = Path(__file__).resolve().parent
QUERIES_PATH = EVAL_DIR / "queries.jsonl"
RESULTS_JSON_PATH = EVAL_DIR / "results.json"
RESULTS_MD_PATH = EVAL_DIR / "results.md"
RESULTS_HTML_PATH = EVAL_DIR / "results.html"


def load_queries(path: Path = QUERIES_PATH) -> list[dict]:
    rows = []
    seen = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "query", "relevant_ids"}
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"{path.name}:{line_number} is missing {sorted(missing)}"
            )
        if row["id"] in seen:
            raise ValueError(f"Duplicate query id {row['id']!r}")
        if not isinstance(row["relevant_ids"], list):
            raise ValueError(f"{row['id']}: relevant_ids must be a list")
        seen.add(row["id"])
        rows.append(row)
    if not rows:
        raise ValueError(f"No evaluation queries in {path}")
    return rows


def _evaluate_retrieval(queries, catalog, store) -> tuple[dict, list[dict]]:
    config_rows: dict[str, list[dict]] = {config: [] for config in CONFIGS}
    per_query = []
    max_k = max(DEFAULT_KS)
    for query_row in queries:
        query_result: dict[str, Any] = {
            "id": query_row["id"],
            "query": query_row["query"],
            "notes": query_row.get("notes", ""),
            "relevant_ids": query_row["relevant_ids"],
            "configs": {},
        }
        for config in CONFIGS:
            retrieved = retrieve(
                store, catalog, query_row["query"], config, k=max_k
            )
            retrieved_ids = [result.id for result in retrieved]
            metric_values = metrics_for_query(
                query_row["relevant_ids"], retrieved_ids
            )
            query_result["configs"][config] = {
                "retrieved_ids": retrieved_ids,
                **metric_values,
            }
            config_rows[config].append(
                {
                    "relevant_ids": query_row["relevant_ids"],
                    "retrieved_ids": retrieved_ids,
                }
            )
        per_query.append(query_result)
    summaries = {
        config: summarize(rows) for config, rows in config_rows.items()
    }
    return summaries, per_query


def _load_api_key() -> str:
    root = EVAL_DIR.parent
    load_dotenv(root / ".env")
    load_dotenv(root / "leasegpt" / ".env")
    return (os.environ.get("GROQ_API_KEY") or "").strip()


def _evaluate_judge(
    queries, catalog, store, api_key: str, existing_rows: list[dict] | None = None
) -> list[dict]:
    existing_by_id = {
        row["id"]: row
        for row in (existing_rows or [])
        if row.get("faithfulness") is not None and not row.get("error")
    }
    rows = []
    for query_row in queries:
        if query_row["id"] in existing_by_id:
            rows.append(existing_by_id[query_row["id"]])
            continue
        result = None
        for attempt in range(3):
            try:
                result = evaluate_generation(
                    query_row["query"], store, catalog, api_key
                )
                break
            except Exception as exc:
                if type(exc).__name__ == "RateLimitError" and attempt < 2:
                    time.sleep(10 * (attempt + 1))
                    continue
                result = {
                    "answer": "",
                    "context_ids": [],
                    "faithfulness": None,
                    "relevance": None,
                    "rationale": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                break
        assert result is not None
        rows.append(
            {
                "id": query_row["id"],
                "query": query_row["query"],
                **result,
            }
        )
    return rows


def _score_average(rows: list[dict], field: str) -> float | None:
    values = [row[field] for row in rows if row.get(field) is not None]
    return mean(values) if values else None


def build_results(skip_llm: bool = False, resume: bool = False) -> dict:
    queries = load_queries()
    catalog = load_catalog()
    validate_relevant_ids(queries, catalog)
    store = build_eval_store(catalog)
    summaries, retrieval_rows = _evaluate_retrieval(queries, catalog, store)

    api_key = "" if skip_llm else _load_api_key()
    existing_rows = []
    if resume and RESULTS_JSON_PATH.is_file():
        existing = json.loads(RESULTS_JSON_PATH.read_text(encoding="utf-8"))
        existing_rows = existing.get("generation_queries", [])
    judge_rows = (
        _evaluate_judge(queries, catalog, store, api_key, existing_rows)
        if api_key
        else []
    )
    if skip_llm:
        judge_status = "skipped"
    elif not api_key:
        judge_status = "missing_api_key"
    else:
        judge_status = "ran"

    return {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "snapshot_fetched_at": SNAPSHOT_FETCHED_AT,
            "snapshot_digest": SNAPSHOT_DIGEST,
            "embedding_model": FASTEMBED_MODEL,
            "query_count": len(queries),
            "ks": list(DEFAULT_KS),
            "configs": list(CONFIGS),
            "rerank_pool_size": 25,
            "judge_model": "openai/gpt-oss-20b",
            "judge_status": judge_status,
        },
        "rubric": JUDGE_RUBRIC,
        "retrieval_summary": summaries,
        "retrieval_queries": retrieval_rows,
        "generation_summary": {
            "judged_query_count": sum(
                row.get("faithfulness") is not None for row in judge_rows
            ),
            "faithfulness_average": _score_average(
                judge_rows, "faithfulness"
            ),
            "relevance_average": _score_average(judge_rows, "relevance"),
        },
        "generation_queries": judge_rows,
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}/5"


def render_markdown(results: dict) -> str:
    metadata = results["metadata"]
    summaries = results["retrieval_summary"]
    lines = [
        "# LeaseGPT RAG Evaluation",
        "",
        (
            f"Generated `{metadata['generated_at']}` against snapshot "
            f"`{metadata['snapshot_fetched_at']}` "
            f"(`{metadata['snapshot_digest'][:12]}`), using "
            f"`{metadata['embedding_model']}`."
        ),
        "",
        "## Method",
        "",
        (
            f"{metadata['query_count']} realistic queries were labeled with "
            "snapshot listing ids. Queries with no relevant ids are qualitative "
            "hard cases and are excluded from macro precision/recall."
        ),
        (
            "The baseline is the app-equivalent dense FAISS ranking. "
            f"The comparison retrieves {metadata['rerank_pool_size']} dense "
            "candidates and re-ranks explicit bedroom, price, neighborhood, "
            "and property-type constraints."
        ),
        (
            "A chunk-size comparison was not used because every current "
            "listing is already one document; changing the 1,000-character "
            "split size would not alter this corpus."
        ),
        "",
        "## Retrieval summary",
        "",
        "| Configuration | Recall@3 | Precision@3 | Recall@5 | Precision@5 | Recall@10 | Precision@10 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config in metadata["configs"]:
        summary = summaries[config]
        lines.append(
            f"| `{config}` | {_percent(summary['recall@3'])} | "
            f"{_percent(summary['precision@3'])} | "
            f"{_percent(summary['recall@5'])} | "
            f"{_percent(summary['precision@5'])} | "
            f"{_percent(summary['recall@10'])} | "
            f"{_percent(summary['precision@10'])} |"
        )

    baseline = summaries["baseline"]["recall@5"]
    reranked = summaries["constraint_rerank"]["recall@5"]
    lines.extend(
        [
            "",
            "## Findings",
            "",
            (
                f"Constraint re-ranking changed mean recall@5 from "
                f"{_percent(baseline)} to {_percent(reranked)} "
                f"({reranked - baseline:+.1%})."
            ),
        ]
    )
    generation = results["generation_summary"]
    if generation["judged_query_count"]:
        lines.append(
            f"Across {generation['judged_query_count']} judged queries, "
            f"faithfulness averaged {_score(generation['faithfulness_average'])} "
            f"and relevance averaged {_score(generation['relevance_average'])}."
        )
    else:
        lines.append(
            "Generation judging was not run. Set `GROQ_API_KEY` and rerun "
            "`uv run python -m eval.run_eval` to add those scores."
        )

    lines.extend(
        [
            "",
            "## Judge rubric",
            "",
            "```text",
            results["rubric"],
            "```",
            "",
            "## Per-query retrieval",
            "",
            "| ID | Query | Gold | Baseline hits@5 | Re-ranked hits@5 |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in results["retrieval_queries"]:
        query = row["query"].replace("|", "\\|")
        baseline_row = row["configs"]["baseline"]
        reranked_row = row["configs"]["constraint_rerank"]
        lines.append(
            f"| {row['id']} | {query} | {len(row['relevant_ids'])} | "
            f"{baseline_row['hits@5'] if baseline_row['hits@5'] is not None else '—'} | "
            f"{reranked_row['hits@5'] if reranked_row['hits@5'] is not None else '—'} |"
        )

    if results["generation_queries"]:
        lines.extend(
            [
                "",
                "## Per-query generation",
                "",
                "| ID | Faithfulness | Relevance | Rationale |",
                "| --- | ---: | ---: | --- |",
            ]
        )
        for row in results["generation_queries"]:
            rationale = (row.get("rationale") or row.get("error") or "").replace(
                "|", "\\|"
            )
            lines.append(
                f"| {row['id']} | {row.get('faithfulness') or '—'} | "
                f"{row.get('relevance') or '—'} | {rationale} |"
            )
    return "\n".join(lines) + "\n"


def write_results(results: dict) -> None:
    RESULTS_JSON_PATH.write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    RESULTS_MD_PATH.write_text(render_markdown(results), encoding="utf-8")
    RESULTS_HTML_PATH.write_text(render_html(results), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Run retrieval evaluation without generation or judge API calls.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse successful judge rows and retry missing or failed rows.",
    )
    args = parser.parse_args()
    results = build_results(skip_llm=args.skip_llm, resume=args.resume)
    write_results(results)
    print(
        f"Wrote {RESULTS_JSON_PATH}, {RESULTS_MD_PATH}, and {RESULTS_HTML_PATH}"
    )


if __name__ == "__main__":
    main()
