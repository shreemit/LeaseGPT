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
ADD_LISTINGS = 1000


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


def merge_listings(existing: list, incoming: list) -> tuple[list, int]:
    """Keep existing order; append incoming records whose ids are new."""
    seen = {_listing_id(item) for item in existing if _listing_id(item)}
    added = []
    for item in incoming:
        listing_id = _listing_id(item)
        if not listing_id or listing_id in seen:
            continue
        seen.add(listing_id)
        added.append(item)
    return existing + added, len(added)


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


def fetch_additional(api_key: str, existing_ids: set[str], add_count: int) -> tuple[list, int]:
    incoming = []
    offset = 0
    pages = 0
    seen = set(existing_ids)
    while len(incoming) < add_count:
        page = fetch_page(api_key, offset, PAGE_SIZE)
        pages += 1
        if not page:
            break
        for record in page:
            listing_id = _listing_id(record)
            if not listing_id or listing_id in seen:
                continue
            seen.add(listing_id)
            incoming.append(record)
            if len(incoming) >= add_count:
                break
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
    return incoming, pages


def main() -> None:
    _load_dotenv()
    existing = _load_existing()
    existing_ids = {_listing_id(item) for item in existing if _listing_id(item)}
    incoming, pages = fetch_additional(_api_key(), existing_ids, ADD_LISTINGS)
    listings, added = merge_listings(existing, incoming)
    LISTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(LISTINGS_PATH, listings)
    write_meta(
        META_PATH,
        {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "query": {
                **QUERY,
                "limit": PAGE_SIZE,
                "added": ADD_LISTINGS,
            },
            "n_listings": len(listings),
        },
    )
    if LEGACY_JSON_PATH.is_file():
        LEGACY_JSON_PATH.unlink()
    print(
        f"Wrote {len(listings)} listings to {LISTINGS_PATH} "
        f"({len(existing)} kept, {added} added, {pages} API pages)"
    )


if __name__ == "__main__":
    main()
