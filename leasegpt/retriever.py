import json
from pathlib import Path
from typing import List

from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from leasegpt.listings import SAMPLE_LISTINGS, SNAPSHOT_FETCHED_AT

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "faiss"

_vector_store = None


class FastEmbedEmbeddings(Embeddings):
    """ONNX embeddings via FastEmbed — no PyTorch, no API key."""

    def __init__(self, model_name: str = FASTEMBED_MODEL):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(self._model.query_embed(text)).tolist()


def get_text_chunks(selection: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, length_function=len
    )
    chunks = []
    for listing in SAMPLE_LISTINGS:
        doc = listing.raw
        if len(doc) > 1200:
            chunks.extend(text_splitter.split_text(doc))
        else:
            chunks.append(doc)
    return chunks


def _index_stamp():
    return {
        "fetched_at": SNAPSHOT_FETCHED_AT,
        "n_listings": len(SAMPLE_LISTINGS),
        "model": FASTEMBED_MODEL,
    }


def _index_is_current() -> bool:
    stamp_path = INDEX_DIR / "stamp.json"
    faiss_path = INDEX_DIR / "index.faiss"
    pkl_path = INDEX_DIR / "index.pkl"
    if not (stamp_path.is_file() and faiss_path.is_file() and pkl_path.is_file()):
        return False
    try:
        saved = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return saved == _index_stamp()


def build_vector_store(chunks=None, force: bool = False):
    """Build or load the FAISS index for the current listing snapshot."""
    global _vector_store
    if chunks is None:
        chunks = get_text_chunks("")
    embeddings = FastEmbedEmbeddings()
    if not force and _index_is_current():
        _vector_store = FAISS.load_local(str(INDEX_DIR), embeddings)
        return _vector_store
    _vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    _vector_store.save_local(str(INDEX_DIR))
    (INDEX_DIR / "stamp.json").write_text(
        json.dumps(_index_stamp(), indent=2) + "\n", encoding="utf-8"
    )
    return _vector_store


def get_set_vector_store(chunks, selection):
    global _vector_store
    if _vector_store is None:
        _vector_store = build_vector_store(chunks=chunks, force=False)
    return _vector_store


def _match_listing(chunk: str):
    snippet = (chunk or "").strip()
    if not snippet:
        return None
    needle = snippet[:80]
    for listing in SAMPLE_LISTINGS:
        if needle in listing.raw or listing.title[:40] in snippet:
            return listing
    return None


def retrieve_sources(vector_store, query: str, k: int = 4):
    """Display-only similarity search. Does not change the generation chain."""
    docs = vector_store.similarity_search(query, k=k)
    sources = []
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else str(doc)
        text = text.strip()
        preview = text[:240] + ("…" if len(text) > 240 else "")
        listing = _match_listing(text)
        sources.append(
            {
                "id": listing.id if listing else None,
                "text": text,
                "preview": preview,
                "title": listing.title if listing else "Retrieved chunk",
                "cost": listing.cost if listing else None,
                "neighborhood": listing.neighborhood if listing else None,
            }
        )
    return sources
