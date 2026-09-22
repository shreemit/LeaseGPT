#!/usr/bin/env python3
"""Fetch a Seattle rental snapshot from RentCast. Not imported by the app."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
LISTINGS_PATH = ROOT / "data" / "seattle_rentals.jsonl"
META_PATH = ROOT / "data" / "seattle_rentals.meta.json"
LEGACY_JSON_PATH = ROOT / "data" / "seattle_rentals.json"
ENDPOINT = "https://api.rentcast.io/v1/listings/rental/long-term"
QUERY = {
    "city": "Seattle",
    "state": "WA",
    "status": "Active",
}
CONTACT_KEYS = ("phone", "email", "website")
PAGE_SIZE = 500
MAX_LISTINGS = 2000


def _load_dotenv() -> None:
    for env_path in (ROOT / ".env", ROOT / "leasegpt" / ".env"):
        if not env_path.is_file():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def _api_key() -> str:
    key = (os.environ.get("RENTCAST_API_KEY") or "").strip()
    if not key:
        raise SystemExit(
            "RENTCAST_API_KEY is missing. Set it in the environment or a gitignored .env."
        )
    return key


def _redact_contact(block):
    if not isinstance(block, dict):
        return block
    cleaned = dict(block)
    for field in CONTACT_KEYS:
        cleaned.pop(field, None)
    return cleaned


def _redact_listing(record: dict) -> dict:
    cleaned = {k: v for k, v in record.items() if k != "history"}
    if "listingAgent" in cleaned:
        cleaned["listingAgent"] = _redact_contact(cleaned["listingAgent"])
    if "listingOffice" in cleaned:
        cleaned["listingOffice"] = _redact_contact(cleaned["listingOffice"])
    return cleaned


def _listing_id(record: dict) -> str:
    return str(record.get("id") or record.get("formattedAddress") or "")


def read_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(item)
    return records


def write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(item, separators=(",", ":"), ensure_ascii=False) for item in records
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_meta(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _parse_records(payload) -> list:
    if isinstance(payload, dict) and "listings" in payload:
        records = payload["listings"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise SystemExit(f"Unexpected RentCast payload type: {type(payload).__name__}")
    return [_redact_listing(item) for item in records if isinstance(item, dict)]


def _load_existing() -> list:
    if LISTINGS_PATH.is_file():
        return read_jsonl(LISTINGS_PATH)
    if LEGACY_JSON_PATH.is_file():
        data = json.loads(LEGACY_JSON_PATH.read_text(encoding="utf-8"))
        records = data.get("listings") or []
        return [item for item in records if isinstance(item, dict)]
    return []


def merge_listings(
    existing: list, incoming: list, *, drop_missing: bool = False
) -> tuple[list, dict]:
    """Update matching ids in place, append new ids, optionally drop unseen ids.

    Keeps existing JSONL order so refreshes stay line-level diffs. Incoming
    records with no id are ignored. When drop_missing is True (full API walk),
    existing rows whose ids never appeared in incoming are removed.
    """
    incoming_by_id = {}
    incoming_order = []
    for item in incoming:
        listing_id = _listing_id(item)
        if not listing_id:
            continue
        if listing_id not in incoming_by_id:
            incoming_order.append(listing_id)
        incoming_by_id[listing_id] = item

    merged = []
    seen = set()
    updated = 0
    for item in existing:
        listing_id = _listing_id(item)
        if listing_id in incoming_by_id:
            fresh = incoming_by_id[listing_id]
            merged.append(fresh)
            seen.add(listing_id)
            if fresh != item:
                updated += 1
        elif listing_id and not drop_missing:
            merged.append(item)
            seen.add(listing_id)

    added = 0
    for listing_id in incoming_order:
        if listing_id in seen:
            continue
        merged.append(incoming_by_id[listing_id])
        seen.add(listing_id)
        added += 1

    dropped = 0
    if drop_missing:
        dropped = sum(
            1 for item in existing if _listing_id(item) not in incoming_by_id
        )
    return merged, {
        "updated": updated,
        "added": added,
        "dropped": dropped,
        "kept": len(merged) - added,
    }


def fetch_page(api_key: str, offset: int, limit: int) -> list:
    params = dict(QUERY)
    params["limit"] = str(limit)
    params["offset"] = str(offset)
    url = f"{ENDPOINT}?{urlencode(params)}"
    request = Request(url, headers={"X-Api-Key": api_key, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        hint = ""
        if exc.code == 403 and "subscription" in body.lower():
            hint = (
                " Activate a plan (the free 50-request tier is enough) at "
                "https://app.rentcast.io/app/api"
            )
        raise SystemExit(f"RentCast HTTP {exc.code}: {body}{hint}") from exc
    except URLError as exc:
        raise SystemExit(f"RentCast request failed: {exc.reason}") from exc
    return _parse_records(payload)


def fetch_snapshot(api_key: str, max_listings: int) -> tuple[list, int, bool]:
    """Page through Active Seattle listings. complete is True if the API ended."""
    incoming = []
    offset = 0
    pages = 0
    seen: set[str] = set()
    complete = False
    while len(incoming) < max_listings:
        page = fetch_page(api_key, offset, PAGE_SIZE)
        pages += 1
        if not page:
            complete = True
            break
        stopped_early = False
        for record in page:
            listing_id = _listing_id(record)
            if not listing_id or listing_id in seen:
                continue
            seen.add(listing_id)
            incoming.append(record)
            if len(incoming) >= max_listings:
                stopped_early = True
                break
        if len(page) < PAGE_SIZE and not stopped_early:
            complete = True
            break
        if stopped_early:
            break
        offset += PAGE_SIZE
    return incoming, pages, complete


def main() -> None:
    _load_dotenv()
    existing = _load_existing()
    incoming, pages, complete = fetch_snapshot(_api_key(), MAX_LISTINGS)
    listings, stats = merge_listings(existing, incoming, drop_missing=complete)
    LISTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(LISTINGS_PATH, listings)
    write_meta(
        META_PATH,
        {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": {
                **QUERY,
                "limit": PAGE_SIZE,
                "max_listings": MAX_LISTINGS,
                "complete": complete,
            },
            "n_listings": len(listings),
        },
    )
    if LEGACY_JSON_PATH.is_file():
        LEGACY_JSON_PATH.unlink()
    print(
        f"Wrote {len(listings)} listings to {LISTINGS_PATH} "
        f"({stats['updated']} updated, {stats['added']} added, "
        f"{stats['dropped']} dropped, {pages} API pages"
        f"{', complete' if complete else ', partial — unseen ids kept'})"
    )


if __name__ == "__main__":
    main()
