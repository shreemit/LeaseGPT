"""Retriever helper tests. Default path uses no FastEmbed, Groq, or network.

Chunking and retrieve_sources run against a tiny in-memory / stub index.
Set LEASEGPT_TEST_FASTEMBED=1 to also exercise a real FastEmbed embed (ONNX
weights download on first run; skip in CI if that is too slow).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS

from leasegpt.listings import SAMPLE_LISTINGS, SNAPSHOT_FETCHED_AT
from leasegpt import retriever as retriever_mod
from leasegpt.retriever import (
    FASTEMBED_MODEL,
    _index_is_current,
    _index_stamp,
    _match_listing,
    get_text_chunks,
    retrieve_sources,
)


class HashEmbeddings(Embeddings):
    """Deterministic bag-of-tokens vectors so FAISS works without FastEmbed."""

    def __init__(self, dim: int = 32):
        self.dim = dim

    def embed_documents(self, texts):
        return [self._vec(text) for text in texts]

    def embed_query(self, text):
        return self._vec(text)

    def _vec(self, text: str):
        vec = [0.0] * self.dim
        for token in text.lower().replace(",", " ").replace("—", " ").split():
            if not token:
                continue
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            vec[int(digest, 16) % self.dim] += 1.0
        norm = math.sqrt(sum(value * value for value in vec)) or 1.0
        return [value / norm for value in vec]


class _StubStore:
    def __init__(self, docs):
        self._docs = docs

    def similarity_search(self, query, k=4):
        return self._docs[:k]


class ChunkingTests(unittest.TestCase):
    def test_snapshot_chunks_keep_listing_text(self):
        chunks = get_text_chunks("")
        self.assertGreaterEqual(len(chunks), len(SAMPLE_LISTINGS))
        self.assertTrue(all(isinstance(chunk, str) and chunk.strip() for chunk in chunks))
        joined = "\n".join(chunks)
        for listing in SAMPLE_LISTINGS[:8]:
            self.assertIn(listing.title, joined)
            self.assertIn(f"Cost: {listing.cost}", joined)

    def test_long_listing_is_split(self):
        long_raw = ("UniqueTokenForSplit " * 80) + ("x" * 400)
        fake = SimpleNamespace(raw=long_raw, title="long", cost=1, neighborhood="X")
        self.assertGreater(len(long_raw), 1200)
        with patch.object(retriever_mod, "SAMPLE_LISTINGS", [fake]):
            chunks = get_text_chunks("ignored-selection")
        self.assertGreater(len(chunks), 1)
        self.assertTrue(any("UniqueTokenForSplit" in chunk for chunk in chunks))

    def test_short_listing_stays_one_chunk(self):
        short = SAMPLE_LISTINGS[0]
        self.assertLessEqual(len(short.raw), 1200)
        with patch.object(retriever_mod, "SAMPLE_LISTINGS", [short]):
            chunks = get_text_chunks("")
        self.assertEqual(chunks, [short.raw])


class MatchAndSourcesTests(unittest.TestCase):
    def test_match_listing_by_title_or_raw_prefix(self):
        listing = SAMPLE_LISTINGS[0]
        self.assertIs(retriever_mod._match_listing(listing.raw), listing)
        self.assertIs(retriever_mod._match_listing(listing.title), listing)
        self.assertIsNone(_match_listing(""))
        self.assertIsNone(_match_listing("no such listing text"))

    def test_retrieve_sources_attaches_listing_metadata(self):
        listing = next(
            item for item in SAMPLE_LISTINGS if item.neighborhood == "U-District"
        )
        store = _StubStore([SimpleNamespace(page_content=listing.raw)])
        sources = retrieve_sources(store, "U-District near UW", k=1)
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(source["title"], listing.title)
        self.assertEqual(source["cost"], listing.cost)
        self.assertEqual(source["neighborhood"], listing.neighborhood)
        self.assertEqual(source["text"], listing.raw.strip())
        self.assertTrue(source["preview"])
        self.assertLessEqual(len(source["preview"]), 241)

    def test_retrieve_sources_fallback_when_chunk_is_unmatched(self):
        store = _StubStore(
            [SimpleNamespace(page_content=""), SimpleNamespace(page_content="orphan chunk")]
        )
        sources = retrieve_sources(store, "anything", k=2)
        self.assertEqual(len(sources), 2)
        self.assertEqual(sources[0]["title"], "Retrieved chunk")
        self.assertIsNone(sources[0]["cost"])
        self.assertEqual(sources[1]["title"], "Retrieved chunk")
        self.assertEqual(sources[1]["text"], "orphan chunk")

    def test_retrieve_sources_honors_k(self):
        docs = [
            SimpleNamespace(page_content=item.raw) for item in SAMPLE_LISTINGS[:5]
        ]
        sources = retrieve_sources(_StubStore(docs), "Seattle", k=3)
        self.assertEqual(len(sources), 3)


class InMemoryIndexTests(unittest.TestCase):
    def test_faiss_hash_index_returns_query_neighborhood(self):
        by_name = {}
        for listing in SAMPLE_LISTINGS:
            if listing.neighborhood in ("U-District", "Ballard", "Downtown"):
                by_name.setdefault(listing.neighborhood, listing)
            if len(by_name) == 3:
                break
        self.assertEqual(len(by_name), 3)
        texts = [item.raw for item in by_name.values()]
        target = by_name["U-District"]
        store = FAISS.from_texts(texts, embedding=HashEmbeddings())
        # Query with the indexed document itself so hash embeddings are exact.
        sources = retrieve_sources(store, target.raw, k=1)
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["neighborhood"], "U-District")
        self.assertEqual(sources[0]["title"], target.title)


class IndexStampTests(unittest.TestCase):
    def test_stamp_tracks_snapshot_count_and_model(self):
        stamp = _index_stamp()
        self.assertEqual(stamp["fetched_at"], SNAPSHOT_FETCHED_AT)
        self.assertEqual(stamp["n_listings"], len(SAMPLE_LISTINGS))
        self.assertEqual(stamp["model"], FASTEMBED_MODEL)

    def test_index_is_current_requires_matching_stamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            index_dir = Path(tmp)
            with patch.object(retriever_mod, "INDEX_DIR", index_dir):
                self.assertFalse(_index_is_current())
                (index_dir / "index.faiss").write_bytes(b"x")
                (index_dir / "index.pkl").write_bytes(b"x")
                (index_dir / "stamp.json").write_text(
                    json.dumps({"fetched_at": "nope", "n_listings": 0, "model": "x"}),
                    encoding="utf-8",
                )
                self.assertFalse(_index_is_current())
                (index_dir / "stamp.json").write_text(
                    json.dumps(_index_stamp()), encoding="utf-8"
                )
                self.assertTrue(_index_is_current())


@unittest.skipUnless(
    os.environ.get("LEASEGPT_TEST_FASTEMBED") == "1",
    "embedding-heavy: set LEASEGPT_TEST_FASTEMBED=1 to run FastEmbed (may download ONNX weights)",
)
class FastEmbedGatedTests(unittest.TestCase):
    def test_fastembed_embeds_a_short_query(self):
        vectors = retriever_mod.FastEmbedEmbeddings().embed_documents(
            ["U-District studio"]
        )
        self.assertEqual(len(vectors), 1)
        self.assertGreater(len(vectors[0]), 8)
        self.assertTrue(all(isinstance(value, float) for value in vectors[0]))


if __name__ == "__main__":
    unittest.main()
