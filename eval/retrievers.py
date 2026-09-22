"""Private evaluation retrievers that never touch the app's FAISS cache."""

from dataclasses import dataclass
import re

from langchain.vectorstores import FAISS

from eval.catalog import EvalListing
from leasegpt.retriever import FastEmbedEmbeddings


BASELINE = "baseline"
CONSTRAINT_RERANK = "constraint_rerank"
CONFIGS = (BASELINE, CONSTRAINT_RERANK)
RERANK_POOL_SIZE = 25


@dataclass(frozen=True)
class RetrievedListing:
    id: str
    text: str
    title: str
    baseline_rank: int
    rerank_score: int = 0


@dataclass(frozen=True)
class QueryConstraints:
    bedrooms: float | int | None = None
    max_price: int | None = None
    min_price: int | None = None
    neighborhoods: tuple[str, ...] = ()
    property_type: str | None = None


def build_eval_store(catalog: list[EvalListing]) -> FAISS:
    """Build an in-memory store with ids in metadata for exact scoring."""
    return FAISS.from_texts(
        [listing.raw for listing in catalog],
        embedding=FastEmbedEmbeddings(),
        metadatas=[
            {"id": listing.id, "title": listing.title} for listing in catalog
        ],
    )


def parse_constraints(query: str, neighborhoods: set[str]) -> QueryConstraints:
    normalized = query.lower().replace(",", "")
    bedrooms: float | int | None = None
    if re.search(r"\bstudio\b", normalized):
        bedrooms = 0
    else:
        bed_match = re.search(r"\b(\d+)\s*(?:bed|bedroom|br)\b", normalized)
        if bed_match:
            bedrooms = int(bed_match.group(1))

    max_price = None
    max_match = re.search(
        r"(?:under|below|less than|up to|max(?:imum)?(?: of)?)\s*\$?(\d{3,5})",
        normalized,
    )
    if max_match:
        max_price = int(max_match.group(1))

    min_price = None
    min_match = re.search(
        r"(?:over|above|more than|at least|min(?:imum)?(?: of)?)\s*\$?(\d{3,5})",
        normalized,
    )
    if min_match:
        min_price = int(min_match.group(1))

    mentioned_neighborhoods = tuple(
        sorted(
            neighborhood
            for neighborhood in neighborhoods
            if neighborhood.lower() in normalized
        )
    )

    property_type = next(
        (
            candidate
            for candidate in ("apartment", "condo", "townhouse", "single family")
            if candidate in normalized
        ),
        None,
    )
    return QueryConstraints(
        bedrooms=bedrooms,
        max_price=max_price,
        min_price=min_price,
        neighborhoods=mentioned_neighborhoods,
        property_type=property_type,
    )


def _constraint_score(
    listing: EvalListing, constraints: QueryConstraints
) -> tuple[int, int]:
    score = 0
    violations = 0
    if constraints.bedrooms is not None:
        if listing.bedrooms == constraints.bedrooms:
            score += 4
        else:
            violations += 1
    if constraints.max_price is not None:
        if listing.cost <= constraints.max_price:
            score += 3
        else:
            violations += 1
    if constraints.min_price is not None:
        if listing.cost >= constraints.min_price:
            score += 3
        else:
            violations += 1
    if constraints.neighborhoods:
        if listing.neighborhood in constraints.neighborhoods:
            score += 4
        else:
            violations += 1
    if constraints.property_type:
        if listing.property_type.lower() == constraints.property_type:
            score += 2
        else:
            violations += 1
    return score, violations


def retrieve(
    store: FAISS,
    catalog: list[EvalListing],
    query: str,
    config: str,
    k: int = 10,
) -> list[RetrievedListing]:
    if config not in CONFIGS:
        raise ValueError(f"Unknown retrieval config: {config}")

    pool_size = k if config == BASELINE else max(k, RERANK_POOL_SIZE)
    docs = store.similarity_search(query, k=pool_size)
    baseline = [
        RetrievedListing(
            id=str(doc.metadata["id"]),
            text=doc.page_content,
            title=str(doc.metadata["title"]),
            baseline_rank=rank,
        )
        for rank, doc in enumerate(docs, start=1)
    ]
    if config == BASELINE:
        return baseline[:k]

    by_id = {listing.id: listing for listing in catalog}
    constraints = parse_constraints(
        query, {listing.neighborhood for listing in catalog}
    )
    rescored = []
    for result in baseline:
        score, violations = _constraint_score(by_id[result.id], constraints)
        rescored.append(
            (
                violations,
                -score,
                result.baseline_rank,
                RetrievedListing(
                    id=result.id,
                    text=result.text,
                    title=result.title,
                    baseline_rank=result.baseline_rank,
                    rerank_score=score,
                ),
            )
        )
    rescored.sort(key=lambda item: item[:3])
    return [item[3] for item in rescored[:k]]
