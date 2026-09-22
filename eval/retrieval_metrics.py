"""Interpretable ranking metrics for the labeled retrieval set."""

from collections.abc import Iterable


DEFAULT_KS = (3, 5, 10)


def _unique_top_k(retrieved_ids: Iterable[str], k: int) -> list[str]:
    if k <= 0:
        raise ValueError("k must be positive")
    unique = []
    seen = set()
    for listing_id in retrieved_ids:
        if listing_id in seen:
            continue
        seen.add(listing_id)
        unique.append(listing_id)
        if len(unique) == k:
            break
    return unique


def recall_at_k(
    relevant_ids: Iterable[str], retrieved_ids: Iterable[str], k: int
) -> float | None:
    """Fraction of known-relevant listings found in the unique top-k.

    Queries with no labeled relevant listings return ``None`` and are excluded
    from macro averages. They remain useful as qualitative hard cases.
    """
    relevant = set(relevant_ids)
    if not relevant:
        return None
    retrieved = set(_unique_top_k(retrieved_ids, k))
    return len(relevant & retrieved) / len(relevant)


def precision_at_k(
    relevant_ids: Iterable[str], retrieved_ids: Iterable[str], k: int
) -> float | None:
    """Fraction of k ranking positions occupied by known-relevant listings."""
    relevant = set(relevant_ids)
    if not relevant:
        return None
    retrieved = _unique_top_k(retrieved_ids, k)
    return len(relevant & set(retrieved)) / k


def metrics_for_query(
    relevant_ids: Iterable[str],
    retrieved_ids: Iterable[str],
    ks: Iterable[int] = DEFAULT_KS,
) -> dict[str, float | int | None]:
    relevant = list(relevant_ids)
    retrieved = list(retrieved_ids)
    metrics: dict[str, float | int | None] = {
        "relevant_count": len(set(relevant)),
    }
    for k in ks:
        metrics[f"recall@{k}"] = recall_at_k(relevant, retrieved, k)
        metrics[f"precision@{k}"] = precision_at_k(relevant, retrieved, k)
        metrics[f"hits@{k}"] = (
            len(set(relevant) & set(_unique_top_k(retrieved, k)))
            if relevant
            else None
        )
    return metrics


def summarize(
    rows: Iterable[dict], ks: Iterable[int] = DEFAULT_KS
) -> dict[str, float | int]:
    """Macro-average retrieval metrics over rows with non-empty gold labels."""
    rows = list(rows)
    labeled = [row for row in rows if row.get("relevant_ids")]
    summary: dict[str, float | int] = {
        "query_count": len(rows),
        "labeled_query_count": len(labeled),
    }
    for k in ks:
        recalls = [
            recall_at_k(row["relevant_ids"], row["retrieved_ids"], k)
            for row in labeled
        ]
        precisions = [
            precision_at_k(row["relevant_ids"], row["retrieved_ids"], k)
            for row in labeled
        ]
        summary[f"recall@{k}"] = (
            sum(value for value in recalls if value is not None) / len(recalls)
            if recalls
            else 0.0
        )
        summary[f"precision@{k}"] = (
            sum(value for value in precisions if value is not None)
            / len(precisions)
            if precisions
            else 0.0
        )
    return summary
