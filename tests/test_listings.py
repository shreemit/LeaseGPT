"""Offline tests for snapshot loading and listing filters. No network."""

import unittest

from leasegpt.listings import (
    NEIGHBORHOODS,
    PRICE_MAX,
    PRICE_MIN,
    SAMPLE_LISTINGS,
    SNAPSHOT_FETCHED_AT,
    SNAPSHOT_LABEL,
    SNAPSHOT_PATH,
    ZIP_TO_NEIGHBORHOOD,
    Listing,
    _bedrooms_label,
    _beds_short,
    _int_price,
    _listing_from_record,
    _load_jsonl,
    _neighborhood,
    _snapshot_date,
    _synthesize_raw,
    filter_listings,
)


class SnapshotLoadTests(unittest.TestCase):
    def test_checked_in_snapshot_maps_to_listings(self):
        self.assertTrue(SNAPSHOT_PATH.is_file())
        records = _load_jsonl(SNAPSHOT_PATH)
        self.assertGreaterEqual(len(records), 12)
        self.assertGreaterEqual(len(SAMPLE_LISTINGS), 12)
        self.assertLessEqual(len(SAMPLE_LISTINGS), len(records))
        self.assertTrue(all(isinstance(item, Listing) for item in SAMPLE_LISTINGS))

    def test_snapshot_label_is_rentcast_not_demo(self):
        self.assertRegex(SNAPSHOT_FETCHED_AT, r"^\d{4}-\d{2}-\d{2}$")
        self.assertIn("RentCast snapshot", SNAPSHOT_LABEL)
        self.assertNotIn("demo snapshot", SNAPSHOT_LABEL)

    def test_listing_raw_is_synthesized_rag_text(self):
        listing = SAMPLE_LISTINGS[0]
        self.assertTrue(listing.raw.startswith("Title:"))
        self.assertIn("Cost:", listing.raw)
        self.assertIn("Neighborhood:", listing.raw)
        self.assertGreater(listing.cost, 0)
        self.assertTrue(listing.title)
        self.assertTrue(listing.neighborhood)

    def test_neighborhoods_and_price_bounds_come_from_snapshot(self):
        costs = [item.cost for item in SAMPLE_LISTINGS]
        names = {item.neighborhood for item in SAMPLE_LISTINGS}
        self.assertEqual(PRICE_MIN, min(costs))
        self.assertGreater(PRICE_MAX, PRICE_MIN)
        self.assertEqual(NEIGHBORHOODS[0], "All")
        self.assertEqual(NEIGHBORHOODS[1:], sorted(names))
        mapped = set(ZIP_TO_NEIGHBORHOOD.values())
        self.assertTrue(names & mapped)


class FilterListingsTests(unittest.TestCase):
    def test_all_neighborhoods_returns_every_listing_in_band(self):
        results = filter_listings(PRICE_MIN, PRICE_MAX, "All")
        self.assertEqual(results, SAMPLE_LISTINGS)

    def test_neighborhood_filter_is_exact(self):
        results = filter_listings(0, 10_000, "U-District")
        self.assertTrue(results)
        self.assertTrue(all(item.neighborhood == "U-District" for item in results))

    def test_price_filter_is_inclusive(self):
        lo, hi = 1500, 2500
        results = filter_listings(lo, hi, "All")
        self.assertTrue(results)
        self.assertTrue(all(lo <= item.cost <= hi for item in results))
        below = [item for item in SAMPLE_LISTINGS if item.cost < lo]
        above = [item for item in SAMPLE_LISTINGS if item.cost > hi]
        self.assertTrue(below or above)
        self.assertTrue(all(item not in results for item in below + above))

    def test_empty_when_price_band_or_neighborhood_misses(self):
        self.assertEqual(filter_listings(1, 0, "All"), [])
        self.assertEqual(filter_listings(PRICE_MIN, PRICE_MAX, "Not A Neighborhood"), [])


class ListingHelperTests(unittest.TestCase):
    def test_neighborhood_uses_zip_map_or_fallback(self):
        self.assertEqual(_neighborhood("98105"), "U-District")
        self.assertEqual(_neighborhood("98122"), "Central")
        self.assertEqual(_neighborhood(" 98107 "), "Ballard")
        self.assertEqual(_neighborhood("99999"), "Seattle 99999")
        self.assertEqual(_neighborhood(""), "Seattle")
        self.assertEqual(_neighborhood(None), "Seattle")

    def test_bedroom_labels(self):
        self.assertEqual(_bedrooms_label(0), "Studio")
        self.assertEqual(_bedrooms_label(2), "2-bedroom")
        self.assertEqual(_bedrooms_label(None), "Unknown beds")
        self.assertEqual(_bedrooms_label("x"), "Unknown beds")
        self.assertEqual(_beds_short(0), "studio")
        self.assertEqual(_beds_short(3), "3br")
        self.assertEqual(_beds_short(None), "?br")

    def test_int_price_rejects_bad_values(self):
        self.assertEqual(_int_price(2400), 2400)
        self.assertEqual(_int_price("2400"), 2400)
        self.assertIsNone(_int_price(None))
        self.assertIsNone(_int_price("n/a"))

    def test_snapshot_date_parses_iso_or_falls_back(self):
        self.assertEqual(_snapshot_date("2026-09-12T21:19:49Z"), "2026-09-12")
        self.assertEqual(_snapshot_date(""), "unknown date")
        self.assertEqual(_snapshot_date("not-a-date"), "not-a-date")

    def test_listing_from_record_skips_unusable_rows(self):
        self.assertIsNone(_listing_from_record({"formattedAddress": "1 Main", "price": None}))
        self.assertIsNone(_listing_from_record({"formattedAddress": "", "price": 1000}))
        listing = _listing_from_record(
            {
                "formattedAddress": "4707 Brooklyn Ave Ne Apt E, Seattle, WA 98105",
                "price": 2100,
                "zipCode": "98105",
                "bedrooms": 1,
                "bathrooms": 1,
                "squareFootage": 600,
                "propertyType": "Apartment",
                "status": "Active",
                "listedDate": "2026-08-01T12:00:00Z",
                "daysOnMarket": 12,
                "listingOffice": {"name": "Example Realty"},
            }
        )
        self.assertIsNotNone(listing)
        self.assertEqual(listing.cost, 2100)
        self.assertEqual(listing.neighborhood, "U-District")
        self.assertEqual(listing.bedrooms_label, "1-bedroom")
        self.assertIn("4707 Brooklyn Ave Ne Apt E", listing.title)
        self.assertIn("Address: 4707 Brooklyn Ave Ne Apt E, Seattle, WA 98105", listing.raw)
        self.assertIn("ZIP: 98105", listing.raw)
        self.assertIn("Listing office: Example Realty", listing.raw)
        self.assertIn("Listed: 2026-08-01", listing.raw)

    def test_synthesize_raw_omits_empty_optional_fields(self):
        raw = _synthesize_raw(
            {"formattedAddress": "1 Main St", "zipCode": ""},
            title="$1,000 / 1br — 1 Main St",
            cost=1000,
            neighborhood="Downtown",
        )
        self.assertIn("Title: $1,000 / 1br — 1 Main St", raw)
        self.assertIn("Cost: 1000", raw)
        self.assertNotIn("ZIP:", raw)
        self.assertNotIn("Bedrooms:", raw)


if __name__ == "__main__":
    unittest.main()
