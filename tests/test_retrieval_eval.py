import json
import unittest
from pathlib import Path

from leasegpt.listings import SAMPLE_LISTINGS, SNAPSHOT_FETCHED_AT
from leasegpt.retrieval_eval import (
    DEFAULT_GOLD_PATH,
    DEFAULT_THRESHOLDS,
    GoldQuery,
    check_thresholds,
    evaluate_sources,
    listing_ids_from_sources,
    load_gold,
    score_ranked_ids,
)
from leasegpt.retriever import _match_listing

ROOT = Path(__file__).resolve().parent.parent


class GoldSetTests(unittest.TestCase):
    def test_gold_ids_exist_in_snapshot(self):
        queries, payload = load_gold()
        snapshot_ids = {listing.id for listing in SAMPLE_LISTINGS}
        self.assertGreaterEqual(len(queries), DEFAULT_THRESHOLDS["n_queries"])
        self.assertEqual(payload.get("snapshot_fetched_at"), SNAPSHOT_FETCHED_AT)
        self.assertEqual(payload.get("source"), "data/seattle_rentals.jsonl")
        seen_query_ids = set()
        for gold in queries:
            self.assertNotIn(gold.id, seen_query_ids)
            seen_query_ids.add(gold.id)
            self.assertTrue(gold.query)
            for listing_id in gold.relevant_ids:
                self.assertIn(
                    listing_id,
                    snapshot_ids,
                    f"{gold.id} relevant id missing from snapshot: {listing_id}",
                )

    def test_gold_file_is_checked_in_json(self):
        self.assertTrue(DEFAULT_GOLD_PATH.is_file())
        payload = json.loads(DEFAULT_GOLD_PATH.read_text(encoding="utf-8"))
        self.assertIsInstance(payload["queries"], list)
        self.assertGreaterEqual(len(payload["queries"]), 12)


class MetricsTests(unittest.TestCase):
    def test_hit_mrr_and_precision(self):
        hits, precision, recall, mrr, rank = score_ranked_ids(
            ["a", "b", "gold", "c"],
            ["gold"],
            cutoffs=(1, 4, 10),
        )
        self.assertEqual(rank, 3)
        self.assertEqual(mrr, 1 / 3)
        self.assertEqual(hits[1], 0.0)
        self.assertEqual(hits[4], 1.0)
        self.assertEqual(precision[4], 0.25)
        self.assertEqual(recall[4], 1.0)

    def test_miss_is_zero(self):
        hits, precision, recall, mrr, rank = score_ranked_ids(
            ["a", "b"],
            ["gold"],
            cutoffs=(1, 4),
        )
        self.assertIsNone(rank)
        self.assertEqual(mrr, 0.0)
        self.assertEqual(hits[4], 0.0)
        self.assertEqual(precision[4], 0.0)
        self.assertEqual(recall[4], 0.0)

    def test_multi_relevant_precision_and_recall(self):
        hits, precision, recall, mrr, rank = score_ranked_ids(
            ["g1", "x", "g2", "y"],
            ["g1", "g2", "g3"],
            cutoffs=(4,),
        )
        self.assertEqual(rank, 1)
        self.assertEqual(mrr, 1.0)
        self.assertEqual(hits[4], 1.0)
        self.assertEqual(precision[4], 0.5)
        self.assertAlmostEqual(recall[4], 2 / 3)

    def test_listing_ids_dedupe_and_skip_unmatched(self):
        ranked, unmatched = listing_ids_from_sources(
            [
                {"id": "a", "title": "A"},
                {"id": None, "title": "orphan"},
                {"id": "a", "title": "A again"},
                {"id": "b", "title": "B"},
            ]
        )
        self.assertEqual(ranked, ["a", "b"])
        self.assertEqual(unmatched, 1)

    def test_listing_ids_fall_back_to_title(self):
        listing = SAMPLE_LISTINGS[0]
        ranked, unmatched = listing_ids_from_sources(
            [{"id": None, "title": listing.title}],
            listings=SAMPLE_LISTINGS,
        )
        self.assertEqual(ranked, [listing.id])
        self.assertEqual(unmatched, 0)

    def test_evaluate_sources_and_thresholds(self):
        gold = [
            GoldQuery(id="q1", query="first", relevant_ids=("gold",)),
            GoldQuery(id="q2", query="second", relevant_ids=("other",)),
        ]
        canned = {
            "first": [{"id": "gold"}, {"id": "x"}, {"id": "y"}, {"id": "z"}],
            "second": [{"id": "nope"}, {"id": "other"}, {"id": "x"}, {"id": "y"}],
        }

        def retrieve(query: str, k: int):
            return canned[query][:k]

        report = evaluate_sources(gold, retrieve, k=4, snapshot_fetched_at="2026-09-12")
        self.assertEqual(report.n_queries, 2)
        self.assertEqual(report.hit_at_k[4], 1.0)
        self.assertEqual(report.mrr, (1.0 + 0.5) / 2)
        self.assertEqual(
            check_thresholds(
                report,
                {"n_queries": 2, "hit_at_4": 0.90, "mrr": 0.70, "matched_source_rate": 0.95},
            ),
            [],
        )

        failing = evaluate_sources(
            gold,
            lambda query, k: [{"id": "miss"}] * k,
            k=4,
        )
        failures = check_thresholds(failing)
        self.assertTrue(any(item.startswith("hit@4") for item in failures))
        self.assertTrue(any(item.startswith("mrr") for item in failures))


class RetrieverMappingTests(unittest.TestCase):
    def test_sample_listings_have_unique_ids(self):
        ids = [listing.id for listing in SAMPLE_LISTINGS]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(ids))

    def test_match_listing_uses_raw_prefix(self):
        listing = SAMPLE_LISTINGS[0]
        matched = _match_listing(listing.raw)
        self.assertIsNotNone(matched)
        self.assertEqual(matched.id, listing.id)

    def test_app_does_not_import_eval_script(self):
        app_text = (ROOT / "app.py").read_text(encoding="utf-8")
        self.assertNotIn("eval_retrieval", app_text)
        self.assertNotIn("retrieval_eval", app_text)
        self.assertNotIn("api.rentcast.io", app_text)


if __name__ == "__main__":
    unittest.main()
