# LeaseGPT

LeaseGPT is a RAG-based apartment leasing assistant. It retrieves from a small set of Seattle listing texts (FAISS + OpenAI embeddings) and answers as a conversational leasing agent in a Streamlit chat UI.

## Architecture

```
listings (leasegpt/listings.py)
        → retriever (chunk + embed + FAISS)
        → generator (RetrievalQA tool + conversational agent)
        → Streamlit UI (app.py)
```

- **Retriever** (`leasegpt/retriever.py`): splits listing documents, builds or loads a pickled FAISS store (`craigslist_vector_store.pkl`), and exposes similarity search via LangChain.
- **Generator** (`leasegpt/generator.py`): wraps retrieval in a LangChain tool and a `chat-conversational-react-description` agent (`gpt-3.5-turbo`).
- **UI** (`app.py`): Streamlit sidebar (city + OpenAI API key) and chat transcript. City selection is present in the UI; listings are still the hardcoded Seattle sample set.
- **Scraper** (`leasegpt/scraper.py`): standalone Craigslist Selenium script. It is not wired into retrieval yet.

## Setup

Python 3.10 is required (`faiss-cpu==1.7.3` has no wheels for 3.13). Install [uv](https://docs.astral.sh/uv/) if you do not already have it.

1. Sync the virtual environment from the lockfile:

```sh
uv sync
```

2. Start the app:

```sh
uv run streamlit run app.py
```

3. In the sidebar, paste an OpenAI API key. You can also set `OPENAI_API_KEY` in a local `.env` file (not committed).

Firefox/geckodriver is only required if you run `leasegpt/scraper.py` yourself.

## Roadmap

Retrieval evaluation is in progress. The next phase is an evaluation layer over the retriever (grounded listing queries, ranking metrics, and regression checks) before changing generation or scraping.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
