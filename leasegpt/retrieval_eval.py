"""Offline retrieval metrics over the frozen RentCast snapshot.

Does not call Groq, OpenAI, RentCast, or any listings API. The gate script
builds/loads the local FastEmbed + FAISS index used by the app.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
from typing import Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD_PATH = ROOT / "tests" / "eval" / "retrieval_gold.json"
DEFAULT_K = 4
EVAL_K = 10

# Regression floors for the checked-in Seattle snapshot. These are demo-corpus
# checks, not published IR benchmarks.
DEFAULT_THRESHOLDS = {
    "n_queries": 12,
    "hit_at_4": 0.85,
    "mrr": 0.70,
    "matched_source_rate": 0.95,
}


@dataclass(frozen=True)
class GoldQuery:
    id: str
    query: str
    relevant_ids: tuple[str, ...]
    notes: str = ""


@dataclass
class QueryScore:
    gold_id: str
    query: str
    ranked_ids: list[str]
    relevant_ids: tuple[str, ...]
    hit_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    reciprocal_rank: float
    first_relevant_rank: int | None
    n_sources: int
    n_unmatched_sources: int


@dataclass
class EvalReport:
    snapshot_fetched_at: str
    n_queries: int
    k: int
    hit_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    matched_source_rate: float
    per_query: list[QueryScore] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "snapshot_fetched_at": self.snapshot_fetched_at,
            "n_queries": self.n_queries,
            "k": self.k,
            "hit@4": self.hit_at_k.get(4),
            "hit@10": self.hit_at_k.get(10),
            "mrr": self.mrr,
            "precision@4": self.precision_at_k.get(4),
            "recall@4": self.recall_at_k.get(4),
            "matched_source_rate": self.matched_source_rate,
            "per_query": [
                {
                    "id": row.gold_id,
                    "query": row.query,
                    "hit@4": row.hit_at_k.get(4),
                    "first_relevant_rank": row.first_relevant_rank,
                    "reciprocal_rank": row.reciprocal_rank,
                    "ranked_ids": row.ranked_ids,
                    "relevant_ids": list(row.relevant_ids),
                }
                for row in self.per_query
            ],
            "warnings": list(self.warnings),
        }


def load_gold(path: Path | None = None) -> tuple[list[GoldQuery], dict]:
    gold_path = Path(path) if path is not None else DEFAULT_GOLD_PATH
    payload = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("queries"), list):
        raise ValueError(f"Gold file must be an object with a queries list: {gold_path}")
    queries = []
    for index, item in enumerate(payload["queries"]):
        if not isinstance(item, dict):
            raise ValueError(f"Gold query {index} must be an object")
        query_id = str(item.get("id") or "").strip() or f"query-{index}"
        query = str(item.get("query") or "").strip()
        relevant = tuple(
            str(listing_id).strip()
            for listing_id in (item.get("relevant_ids") or [])
            if str(listing_id).strip()
        )
        if not query:
            raise ValueError(f"Gold query {query_id} is missing query text")
        if not relevant:
            raise ValueError(f"Gold query {query_id} has no relevant_ids")
        queries.append(
            GoldQuery(
                id=query_id,
                query=query,
                relevant_ids=relevant,
                notes=str(item.get("notes") or ""),
            )
        )
    return queries, payload


def listing_ids_from_sources(
    sources: Sequence[dict], listings: Iterable | None = None
) -> tuple[list[str], int]:
    """Unique listing ids in retrieval order. Unmatched chunks are skipped."""
    title_to_id = {}
    if listings is not None:
        title_to_id = {
            listing.title: listing.id
            for listing in listings
            if getattr(listing, "id", None) and getattr(listing, "title", None)
        }
    ranked: list[str] = []
    seen: set[str] = set()
    unmatched = 0
    for source in sources:
        listing_id = str(source.get("id") or "").strip()
        if not listing_id:
            listing_id = title_to_id.get(source.get("title") or "", "")
        if not listing_id:
            unmatched += 1
            continue
        if listing_id in seen:
            continue
        seen.add(listing_id)
        ranked.append(listing_id)
    return ranked, unmatched


def first_relevant_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> int | None:
    relevant = set(relevant_ids)
    for index, listing_id in enumerate(ranked_ids, start=1):
        if listing_id in relevant:
            return index
    return None


def score_ranked_ids(
    ranked_ids: Sequence[str],
    relevant_ids: Sequence[str],
    cutoffs: Sequence[int] = (1, 4, 10),
) -> tuple[dict[int, float], dict[int, float], dict[int, float], float, int | None]:
    relevant = set(relevant_ids)
    rank = first_relevant_rank(ranked_ids, relevant_ids)
    reciprocal = 0.0 if rank is None else 1.0 / rank
    hits: dict[int, float] = {}
    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    n_relevant = len(relevant)
    for cutoff in cutoffs:
        top = list(ranked_ids[:cutoff])
        n_hit = sum(1 for listing_id in top if listing_id in relevant)
        hits[cutoff] = 1.0 if n_hit else 0.0
        precision[cutoff] = (n_hit / cutoff) if cutoff else 0.0
        recall[cutoff] = (n_hit / n_relevant) if n_relevant else 0.0
    return hits, precision, recall, reciprocal, rank


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def aggregate_scores(
    rows: Sequence[QueryScore],
    *,
    snapshot_fetched_at: str,
    k: int,
    cutoffs: Sequence[int] = (1, 4, 10),
    warnings: Sequence[str] | None = None,
) -> EvalReport:
    total_sources = sum(row.n_sources for row in rows)
    unmatched = sum(row.n_unmatched_sources for row in rows)
    matched = max(total_sources - unmatched, 0)
    matched_rate = (matched / total_sources) if total_sources else 1.0
    return EvalReport(
        snapshot_fetched_at=snapshot_fetched_at,
        n_queries=len(rows),
        k=k,
        hit_at_k={cutoff: _mean([row.hit_at_k[cutoff] for row in rows]) for cutoff in cutoffs},
        precision_at_k={
            cutoff: _mean([row.precision_at_k[cutoff] for row in rows]) for cutoff in cutoffs
        },
        recall_at_k={
            cutoff: _mean([row.recall_at_k[cutoff] for row in rows]) for cutoff in cutoffs
        },
        mrr=_mean([row.reciprocal_rank for row in rows]),
        matched_source_rate=matched_rate,
        per_query=list(rows),
        warnings=list(warnings or []),
    )


def evaluate_sources(
    gold_queries: Sequence[GoldQuery],
    retrieve: Callable[[str, int], Sequence[dict]],
    *,
    k: int = EVAL_K,
    cutoffs: Sequence[int] = (1, 4, 10),
    snapshot_fetched_at: str = "",
    listings: Iterable | None = None,
) -> EvalReport:
    """Score a retrieve(query, k) -> sources callback. No model imports here."""
    rows: list[QueryScore] = []
    warnings: list[str] = []
    for gold in gold_queries:
        sources = list(retrieve(gold.query, k))
        ranked, unmatched = listing_ids_from_sources(sources, listings=listings)
        if unmatched:
            warnings.append(
                f"{gold.id}: {unmatched} retrieved chunk(s) did not map to a listing id"
            )
        hits, precision, recall, reciprocal, rank = score_ranked_ids(
            ranked, gold.relevant_ids, cutoffs=cutoffs
        )
        rows.append(
            QueryScore(
                gold_id=gold.id,
                query=gold.query,
                ranked_ids=ranked,
                relevant_ids=gold.relevant_ids,
                hit_at_k=hits,
                precision_at_k=precision,
                recall_at_k=recall,
                reciprocal_rank=reciprocal,
                first_relevant_rank=rank,
                n_sources=len(sources),
                n_unmatched_sources=unmatched,
            )
        )
    return aggregate_scores(
        rows,
        snapshot_fetched_at=snapshot_fetched_at,
        k=k,
        cutoffs=cutoffs,
        warnings=warnings,
    )


def check_thresholds(
    report: EvalReport, thresholds: dict | None = None
) -> list[str]:
    floors = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        floors.update(thresholds)
    failures: list[str] = []
    if report.n_queries < floors["n_queries"]:
        failures.append(
            f"n_queries {report.n_queries} < {floors['n_queries']}"
        )
    hit4 = report.hit_at_k.get(4, 0.0)
    if hit4 + 1e-12 < floors["hit_at_4"]:
        failures.append(f"hit@4 {hit4:.3f} < {floors['hit_at_4']:.3f}")
    if report.mrr + 1e-12 < floors["mrr"]:
        failures.append(f"mrr {report.mrr:.3f} < {floors['mrr']:.3f}")
    if report.matched_source_rate + 1e-12 < floors["matched_source_rate"]:
        failures.append(
            f"matched_source_rate {report.matched_source_rate:.3f} "
            f"< {floors['matched_source_rate']:.3f}"
        )
    return failures


def format_report(report: EvalReport, failures: Sequence[str] | None = None) -> str:
    lines = [
        (
            f"Retrieval eval  snapshot={report.snapshot_fetched_at or 'unknown'}  "
            f"n={report.n_queries}  retrieve_k={report.k}"
        ),
        (
            f"hit@1 {report.hit_at_k.get(1, 0.0):.3f}   "
            f"hit@4 {report.hit_at_k.get(4, 0.0):.3f}   "
            f"hit@10 {report.hit_at_k.get(10, 0.0):.3f}"
        ),
        (
            f"MRR   {report.mrr:.3f}   "
            f"P@4 {report.precision_at_k.get(4, 0.0):.3f}   "
            f"R@4 {report.recall_at_k.get(4, 0.0):.3f}   "
            f"matched {report.matched_source_rate:.3f}"
        ),
        "",
        f"{'query':<28} {'hit@4':<7} {'rank':<6} first gold id",
    ]
    for row in report.per_query:
        rank = "-" if row.first_relevant_rank is None else str(row.first_relevant_rank)
        first_gold = row.relevant_ids[0] if row.relevant_ids else ""
        lines.append(
            f"{row.gold_id:<28} {row.hit_at_k.get(4, 0.0):<7.0f} {rank:<6} {first_gold}"
        )
    if report.warnings:
        lines.append("")
        lines.append("warnings:")
        lines.extend(f"- {warning}" for warning in report.warnings)
    lines.append("")
    if failures:
        lines.append("FAIL  " + "; ".join(failures))
    else:
        lines.append(
            "PASS  "
            f"hit@4>={DEFAULT_THRESHOLDS['hit_at_4']:.2f}  "
            f"MRR>={DEFAULT_THRESHOLDS['mrr']:.2f}  "
            f"n>={DEFAULT_THRESHOLDS['n_queries']}"
        )
    lines.append(
        "Note: scores are a regression check on a dated Seattle sample, "
        "not live inventory or a production IR benchmark."
    )
    return "\n".join(lines) + "\n"
