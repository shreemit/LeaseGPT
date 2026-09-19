import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_fetch_module():
    path = ROOT / "scripts" / "fetch_rentcast_listings.py"
    spec = importlib.util.spec_from_file_location("fetch_rentcast_listings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RedactListingTests(unittest.TestCase):
    def test_strips_contacts_and_history(self):
        fetch = _load_fetch_module()
        record = {
            "formattedAddress": "123 Example Ave, Seattle, WA 98105",
            "price": 2400,
            "history": {"2026-01-01": {"price": 2300}},
            "listingAgent": {
                "name": "Alex Agent",
                "phone": "2065550100",
                "email": "alex@example.com",
                "website": "https://example.com",
            },
            "listingOffice": {
                "name": "Example Realty",
                "phone": "2065550199",
                "email": "office@example.com",
            },
        }
        cleaned = fetch._redact_listing(record)
        self.assertNotIn("history", cleaned)
        self.assertEqual(cleaned["listingAgent"], {"name": "Alex Agent"})
        self.assertEqual(cleaned["listingOffice"], {"name": "Example Realty"})
        self.assertEqual(record["listingAgent"]["phone"], "2065550100")

    def test_merge_listings_keeps_existing_and_appends_new(self):
        fetch = _load_fetch_module()
        existing = [{"id": "a", "price": 1}, {"id": "b", "price": 2}]
        incoming = [{"id": "b", "price": 99}, {"id": "c", "price": 3}]
        merged, added = fetch.merge_listings(existing, incoming)
        self.assertEqual([item["id"] for item in merged], ["a", "b", "c"])
        self.assertEqual(merged[1]["price"], 2)
        self.assertEqual(added, 1)

    def test_jsonl_roundtrip(self):
        fetch = _load_fetch_module()
        path = ROOT / "data" / "_test_roundtrip.jsonl"
        records = [{"id": "a", "price": 1}, {"id": "b", "price": 2}]
        try:
            fetch.write_jsonl(path, records)
            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 2)
            self.assertEqual(fetch.read_jsonl(path), records)
        finally:
            if path.is_file():
                path.unlink()

    def test_snapshot_maps_to_listings(self):
        from leasegpt.listings import SAMPLE_LISTINGS, SNAPSHOT_LABEL, filter_listings

        self.assertGreaterEqual(len(SAMPLE_LISTINGS), 12)
        self.assertTrue(SAMPLE_LISTINGS[0].id)
        self.assertTrue(SAMPLE_LISTINGS[0].raw.startswith("Title:"))
        self.assertIn("RentCast snapshot", SNAPSHOT_LABEL)
        self.assertNotIn("demo snapshot", SNAPSHOT_LABEL)
        u_district = filter_listings(0, 10_000, "U-District")
        self.assertTrue(u_district)
        self.assertTrue(all(item.neighborhood == "U-District" for item in u_district))

    def test_app_does_not_import_fetch_script(self):
        app_text = (ROOT / "app.py").read_text(encoding="utf-8")
        retriever_text = (ROOT / "leasegpt" / "retriever.py").read_text(encoding="utf-8")
        listings_text = (ROOT / "leasegpt" / "listings.py").read_text(encoding="utf-8")
        ui_text = (ROOT / "leasegpt" / "ui.py").read_text(encoding="utf-8")
        for text in (app_text, retriever_text, listings_text, ui_text):
            self.assertNotIn("import fetch_rentcast", text)
            self.assertNotIn("from fetch_rentcast", text)
            self.assertNotIn("api.rentcast.io", text)


if __name__ == "__main__":
    unittest.main()
