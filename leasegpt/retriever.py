from typing import List

from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

from leasegpt.listings import SAMPLE_LISTINGS, doc1, doc2, doc3, doc4

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"

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
    # TODO: Scraping Craigslist
    text = " ".join([doc1, doc2, doc3, doc4])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, length_function=len
    )
    docs = [doc1, doc2, doc3, doc4]
    chunks = []

    # Splitting the text into chunks
    for doc in docs:
        if len(doc) > 1200:
            chunk_doc = text_splitter.split_text(doc)
            for chunk in chunk_doc:
                chunks.append(chunk)
        else:
            chunks.append(doc)
    return chunks


def get_set_vector_store(chunks, selection):
    global _vector_store
    if _vector_store is None:
        embeddings = FastEmbedEmbeddings()
        _vector_store = FAISS.from_texts(chunks, embedding=embeddings)
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
                "text": text,
                "preview": preview,
                "title": listing.title if listing else "Retrieved chunk",
                "cost": listing.cost if listing else None,
                "neighborhood": listing.neighborhood if listing else None,
            }
        )
    return sources
