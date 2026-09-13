"""Load the frozen RentCast snapshot with stable ids for evaluation."""

from dataclasses import dataclass
import json
from typing import Iterable

from leasegpt.listings import SNAPSHOT_PATH, _listing_from_record


@dataclass(frozen=True)
class EvalListing:
    id: str
    title: str
    cost: int
    neighborhood: str
    bedrooms: float | int | None
    property_type: str
    raw: str


def load_catalog() -> list[EvalListing]:
    """Return usable snapshot listings while preserving their RentCast ids."""
    listings = []
    seen_ids = set()
    for line_number, line in enumerate(
        SNAPSHOT_PATH.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        listing_id = str(record.get("id") or "").strip()
        listing = _listing_from_record(record)
        if not listing_id or listing is None:
            continue
        if listing_id in seen_ids:
            raise ValueError(
                f"Duplicate listing id {listing_id!r} on snapshot line {line_number}"
            )
        seen_ids.add(listing_id)
        listings.append(
            EvalListing(
                id=listing_id,
                title=listing.title,
                cost=listing.cost,
                neighborhood=listing.neighborhood,
                bedrooms=record.get("bedrooms"),
                property_type=str(record.get("propertyType") or ""),
                raw=listing.raw,
            )
        )
    if not listings:
        raise ValueError(f"No usable evaluation listings in {SNAPSHOT_PATH}")
    return listings


def validate_relevant_ids(
    query_rows: Iterable[dict], catalog: Iterable[EvalListing]
) -> None:
    """Fail fast when a gold label no longer exists in the frozen snapshot."""
    known_ids = {listing.id for listing in catalog}
    missing = {
        relevant_id
        for row in query_rows
        for relevant_id in row.get("relevant_ids", [])
        if relevant_id not in known_ids
    }
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Evaluation labels reference missing listing ids: {preview}")
