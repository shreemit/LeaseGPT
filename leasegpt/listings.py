from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
import json

SNAPSHOT_PATH = Path(__file__).resolve().parent.parent / "data" / "seattle_rentals.jsonl"
META_PATH = Path(__file__).resolve().parent.parent / "data" / "seattle_rentals.meta.json"

ZIP_TO_NEIGHBORHOOD = {
    "98101": "Downtown",
    "98102": "Capitol Hill",
    "98103": "Fremont/Wallingford",
    "98104": "Pioneer Square",
    "98105": "U-District",
    "98106": "Delridge",
    "98107": "Ballard",
    "98108": "Georgetown",
    "98109": "South Lake Union",
    "98112": "Madison Park",
    "98115": "North Seattle",
    "98116": "West Seattle",
    "98117": "Ballard",
    "98118": "Rainier Valley",
    "98119": "Queen Anne",
    "98121": "Belltown",
    "98122": "Central",
    "98125": "Northgate",
    "98126": "West Seattle",
    "98133": "Bitter Lake",
    "98134": "SODO",
    "98136": "West Seattle",
    "98144": "Mount Baker",
    "98146": "White Center",
    "98154": "Downtown",
    "98164": "Downtown",
    "98174": "Downtown",
    "98195": "U-District",
    "98199": "Magnolia",
}


@dataclass(frozen=True)
class Listing:
    title: str
    cost: int
    neighborhood: str
    bedrooms_label: str
    raw: str


def _load_jsonl(path: Path) -> list:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            records.append(item)
    return records


def _load_snapshot() -> dict:
    if not SNAPSHOT_PATH.is_file():
        raise FileNotFoundError(
            f"Missing listing snapshot at {SNAPSHOT_PATH}. "
            "Run: uv run python scripts/fetch_rentcast_listings.py"
        )
    meta = {}
    if META_PATH.is_file():
        loaded = json.loads(META_PATH.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            meta = loaded
    return {**meta, "listings": _load_jsonl(SNAPSHOT_PATH)}


def _snapshot_date(fetched_at: str) -> str:
    if not fetched_at:
        return "unknown date"
    try:
        return datetime.fromisoformat(fetched_at.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        return fetched_at[:10]


def _neighborhood(zip_code: str) -> str:
    zip_code = (zip_code or "").strip()
    if zip_code in ZIP_TO_NEIGHBORHOOD:
        return ZIP_TO_NEIGHBORHOOD[zip_code]
    return f"Seattle {zip_code}" if zip_code else "Seattle"


def _bedrooms_label(bedrooms) -> str:
    if bedrooms is None:
        return "Unknown beds"
    try:
        count = int(bedrooms)
    except (TypeError, ValueError):
        return "Unknown beds"
    if count == 0:
        return "Studio"
    return f"{count}-bedroom"


def _beds_short(bedrooms) -> str:
    if bedrooms is None:
        return "?br"
    try:
        count = int(bedrooms)
    except (TypeError, ValueError):
        return "?br"
    return "studio" if count == 0 else f"{count}br"


def _int_price(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_line(label: str, value) -> str:
    if value is None or value == "":
        return ""
    return f"{label}: {value}\n"


def _synthesize_raw(record: dict, title: str, cost: int, neighborhood: str) -> str:
    bedrooms = record.get("bedrooms")
    office = record.get("listingOffice") or {}
    office_name = office.get("name") if isinstance(office, dict) else None
    listed = record.get("listedDate") or ""
    if isinstance(listed, str) and "T" in listed:
        listed = listed.split("T", 1)[0]
    parts = [
        f"Title: {title}\n",
        f"Cost: {cost}\n",
        f"Neighborhood: {neighborhood}\n",
        _optional_line("Address", record.get("formattedAddress")),
        _optional_line("ZIP", record.get("zipCode")),
        _optional_line("Property type", record.get("propertyType")),
        _optional_line("Bedrooms", bedrooms if bedrooms is not None else None),
        _optional_line("Bathrooms", record.get("bathrooms")),
        _optional_line("Square footage", record.get("squareFootage")),
        _optional_line("Year built", record.get("yearBuilt")),
        _optional_line("Status", record.get("status")),
        _optional_line("Listed", listed),
        _optional_line("Days on market", record.get("daysOnMarket")),
        _optional_line("Listing office", office_name),
    ]
    return "".join(parts).strip() + "\n"


def _listing_from_record(record: dict) -> Listing | None:
    address = (record.get("formattedAddress") or "").strip()
    cost = _int_price(record.get("price"))
    if not address or cost is None:
        return None
    bedrooms = record.get("bedrooms")
    title = f"${cost:,} / {_beds_short(bedrooms)} — {address}"
    neighborhood = _neighborhood(str(record.get("zipCode") or ""))
    return Listing(
        title=title,
        cost=cost,
        neighborhood=neighborhood,
        bedrooms_label=_bedrooms_label(bedrooms),
        raw=_synthesize_raw(record, title, cost, neighborhood),
    )


_SNAPSHOT = _load_snapshot()
SNAPSHOT_DIGEST = sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
SNAPSHOT_FETCHED_AT = _snapshot_date(_SNAPSHOT.get("fetched_at") or "")
if _SNAPSHOT.get("source") == "demo-placeholder":
    SNAPSHOT_LABEL = f"demo snapshot {SNAPSHOT_FETCHED_AT}"
else:
    SNAPSHOT_LABEL = f"RentCast snapshot {SNAPSHOT_FETCHED_AT}"

SAMPLE_LISTINGS = []
for _record in _SNAPSHOT.get("listings") or []:
    if not isinstance(_record, dict):
        continue
    _listing = _listing_from_record(_record)
    if _listing is not None:
        SAMPLE_LISTINGS.append(_listing)

if not SAMPLE_LISTINGS:
    raise ValueError(
        f"No usable listings in {SNAPSHOT_PATH}. "
        "Re-run: uv run python scripts/fetch_rentcast_listings.py"
    )

NEIGHBORHOODS = ["All"] + sorted({listing.neighborhood for listing in SAMPLE_LISTINGS})
PRICE_MIN = min(listing.cost for listing in SAMPLE_LISTINGS)
PRICE_MAX = max(listing.cost for listing in SAMPLE_LISTINGS)
if PRICE_MAX <= PRICE_MIN:
    PRICE_MAX = PRICE_MIN + 1


def filter_listings(price_min: int, price_max: int, neighborhood: str):
    results = []
    for listing in SAMPLE_LISTINGS:
        if listing.cost < price_min or listing.cost > price_max:
            continue
        if neighborhood != "All" and listing.neighborhood != neighborhood:
            continue
        results.append(listing)
    return results
