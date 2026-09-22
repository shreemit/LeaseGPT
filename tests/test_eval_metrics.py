import unittest

from eval.html_report import render_html
from eval.retrieval_metrics import (
    metrics_for_query,
    precision_at_k,
    recall_at_k,
    summarize,
)


class RetrievalMetricTests(unittest.TestCase):
    def test_recall_and_precision_at_k(self):
        relevant = ["a", "c", "e"]
        retrieved = ["a", "b", "c", "d", "e"]

        self.assertEqual(recall_at_k(relevant, retrieved, 3), 2 / 3)
        self.assertEqual(precision_at_k(relevant, retrieved, 3), 2 / 3)

    def test_duplicate_retrievals_do_not_consume_rank(self):
        retrieved = ["a", "a", "b", "c"]

        self.assertEqual(recall_at_k(["a", "c"], retrieved, 3), 1.0)
        self.assertEqual(precision_at_k(["a", "c"], retrieved, 3), 2 / 3)

    def test_unlabeled_hard_case_is_not_scored(self):
        self.assertIsNone(recall_at_k([], ["a"], 3))
        self.assertIsNone(precision_at_k([], ["a"], 3))
        self.assertIsNone(metrics_for_query([], ["a"])["hits@3"])

    def test_summary_macro_averages_only_labeled_queries(self):
        rows = [
            {"relevant_ids": ["a"], "retrieved_ids": ["a", "x", "y"]},
            {"relevant_ids": ["b"], "retrieved_ids": ["x", "y", "z"]},
            {"relevant_ids": [], "retrieved_ids": ["a", "b", "c"]},
        ]

        summary = summarize(rows, ks=(3,))

        self.assertEqual(summary["query_count"], 3)
        self.assertEqual(summary["labeled_query_count"], 2)
        self.assertEqual(summary["recall@3"], 0.5)
        self.assertAlmostEqual(summary["precision@3"], 1 / 6)


class StaticHtmlReportTests(unittest.TestCase):
    def test_renders_charts_and_escapes_query_text(self):
        results = {
            "metadata": {
                "snapshot_fetched_at": "2026-09-12",
                "snapshot_digest": "abcdef1234567890",
                "embedding_model": "test/model",
                "generated_at": "now",
                "query_count": 1,
                "judge_status": "skipped",
                "ks": [3, 5, 10],
                "configs": ["baseline", "constraint_rerank"],
            },
            "retrieval_summary": {
                "baseline": {
                    "recall@3": 0.1,
                    "recall@5": 0.2,
                    "recall@10": 0.3,
                    "precision@3": 0.1,
                    "precision@5": 0.2,
                    "precision@10": 0.3,
                },
                "constraint_rerank": {
                    "recall@3": 0.3,
                    "recall@5": 0.4,
                    "recall@10": 0.5,
                    "precision@3": 0.3,
                    "precision@5": 0.4,
                    "precision@10": 0.5,
                },
            },
            "retrieval_queries": [
                {
                    "id": "q1",
                    "query": "<script>alert(1)</script>",
                    "relevant_ids": ["listing"],
                    "configs": {
                        "baseline": {"recall@5": 0.0, "hits@5": 0},
                        "constraint_rerank": {"recall@5": 1.0, "hits@5": 1},
                    },
                }
            ],
            "generation_summary": {
                "judged_query_count": 0,
                "faithfulness_average": None,
                "relevance_average": None,
            },
            "generation_queries": [],
            "rubric": "A strict rubric",
        }

        html = render_html(results)

        self.assertIn("<!doctype html>", html)
        self.assertIn("What this evaluation tests", html)
        self.assertIn("Recall@k", html)
        self.assertIn("How generation is judged", html)
        self.assertIn("Mean recall", html)
        self.assertIn('data-impact="helped"', html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)
        self.assertNotIn("<script>alert(1)</script>", html)


if __name__ == "__main__":
    unittest.main()
