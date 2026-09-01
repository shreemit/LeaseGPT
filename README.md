---
title: LeaseGPT
emoji: 🚪
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# LeaseGPT

LeaseGPT is a RAG-based apartment leasing assistant. It retrieves from a small set of Seattle listing texts (FAISS + local FastEmbed embeddings) and answers as a conversational leasing agent in a Streamlit chat UI. Chat generation uses the Groq Python SDK (`openai/gpt-oss-20b`).

## Architecture

```
listings (leasegpt/listings.py)
        → retriever (chunk + FastEmbed + FAISS)
        → generator (RetrievalQA tool + conversational agent via Groq)
        → Streamlit UI (app.py + leasegpt/ui.py)
```

- **Retriever** (`leasegpt/retriever.py`): splits listing documents, builds an in-memory FAISS store with FastEmbed (`BAAI/bge-small-en-v1.5`, ONNX, no API key), and exposes similarity search via LangChain. `retrieve_sources` is a display-only search used to show grounding chunks in the UI.
- **Generator** (`leasegpt/generator.py`): wraps retrieval in a LangChain tool and a `chat-conversational-react-description` agent. Chat is `ChatGroq` in `leasegpt/groq_chat.py` (Groq SDK, not OpenAI).
- **UI** (`app.py`, `leasegpt/ui.py`): Centered Streamlit shell with Chat / Sources / Listings tabs. Chat has the empty-state example queries and a compact “Why this answer” expander (listing titles). Sources shows retrieved chunks. Listings holds price/neighborhood filters and sample cards. Sidebar is the Groq API key only.
- **Scraper** (`leasegpt/scraper.py`): standalone Craigslist Selenium script. It is not imported by the app and is not wired into retrieval. Running it launches Firefox at import time.

### Screenshots

Add two captures under [`docs/screenshots/`](docs/screenshots/) for the README or a live demo: `empty-state.png` (suggestions + sample cards) and `grounded-answer.png` (chat + retrieved context + expander open).

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

3. Chat needs a free Groq API key ([console.groq.com](https://console.groq.com)). Paste it in the sidebar, or set `GROQ_API_KEY` in a local `.env` file (not committed). Retrieval and listing cards work without a key.

Firefox/geckodriver is only required if you run `leasegpt/scraper.py` yourself.

## Hosting

This is a Streamlit Python server. **GitHub Pages cannot host it** (static files only).

### Hugging Face Spaces (primary)

Streamlit Spaces use the **Docker** SDK. As of 2026, creating Docker Spaces requires [Hugging Face Pro](https://huggingface.co/pro) (or Team/Enterprise). CPU Basic hardware is $0/hour after that; idle Spaces sleep.

1. Create a Space: SDK **Docker**, hardware **CPU Basic**, public.
2. Settings → Variables and secrets → add secret `GROQ_API_KEY` so visitors can chat without pasting a key (they share that Groq free-tier quota).
3. Push this repo to the Space:

```sh
git remote add space https://huggingface.co/spaces/<user>/leasegpt
git push space HEAD:main
```

The first boot downloads FastEmbed ONNX weights. Later cold starts after sleep are slower.

### Streamlit Community Cloud ($0 fallback)

If you do not want Hugging Face Pro, deploy from GitHub at [share.streamlit.io](https://share.streamlit.io): pick this repo, `app.py`, and set `GROQ_API_KEY` in the app secrets. Community Cloud installs from `requirements.txt`.

## Roadmap

Retrieval evaluation is in progress. The next phase is an evaluation layer over the retriever (grounded listing queries, ranking metrics, and regression checks) before changing generation or scraping.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
