#!/usr/bin/env python3
"""Build the local FAISS index from the RentCast snapshot. Not imported by the app."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from leasegpt.retriever import INDEX_DIR, build_vector_store, get_text_chunks


def main() -> None:
    chunks = get_text_chunks("")
    store = build_vector_store(chunks=chunks, force=True)
    n_vecs = store.index.ntotal
    print(f"Indexed {n_vecs} chunks from {len(chunks)} texts → {INDEX_DIR}")


if __name__ == "__main__":
    main()
