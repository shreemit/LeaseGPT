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

LeaseGPT is a RAG-based apartment leasing assistant. It retrieves from a frozen Seattle RentCast rental snapshot (FAISS + local FastEmbed embeddings) and answers as a conversational leasing agent in a Streamlit chat UI. Chat generation uses the Groq Python SDK (`openai/gpt-oss-20b`). This is a dated sample corpus, not live inventory.

## Architecture

```
RentCast snapshot (data/seattle_rentals.jsonl)
        → listings (leasegpt/listings.py)
        → retriever (chunk + FastEmbed + FAISS)
        → generator (RetrievalQA tool + conversational agent via Groq)
        → Streamlit UI (app.py + leasegpt/ui.py)
```

- **Listings** (`leasegpt/listings.py`): loads `data/seattle_rentals.jsonl` plus `data/seattle_rentals.meta.json` (Active Seattle rentals pulled from RentCast) and synthesizes RAG text. The Streamlit app never calls RentCast. Refresh locally with `RENTCAST_API_KEY` (see below).
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

3. Chat needs a free Groq API key ([console.groq.com](https://console.groq.com)). Paste it in the sidebar, or set `GROQ_API_KEY` in a local `.env` file (not committed). Retrieval and listing cards work without a Groq key.

The checked-in RentCast snapshot (`data/seattle_rentals.jsonl`, with `fetched_at` in `data/seattle_rentals.meta.json`) is enough to run the app. It is a dated sample, not live inventory. Refresh it locally (50 free RentCast requests/month; do not call this from Streamlit):

```sh
RENTCAST_API_KEY=... uv run python scripts/fetch_rentcast_listings.py
uv run python scripts/build_vector_store.py
```

The vector index is local FastEmbed + FAISS. After a listing refresh, rebuild it with the second command (or let the app build on first chat). The index under `data/faiss/` is gitignored.

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

## Retrieval evaluation

An offline eval scores the local FastEmbed + FAISS retriever against a small gold set of listing queries (`tests/eval/retrieval_gold.json`). Relevant ids come from the checked-in Seattle snapshot only. The suite does not call Groq, OpenAI, or RentCast.

```sh
uv run python -m unittest tests/test_retrieval_eval.py
uv run python scripts/eval_retrieval.py
```

The unittest target checks gold-file integrity and ranking math (no embeddings). The script is the release gate: it builds or loads the same in-memory/on-disk index the app uses, reports hit@k / MRR / precision@k / recall@k, and exits non-zero if hit@4, MRR, or chunk-to-listing matching fall below fixed floors in `leasegpt/retrieval_eval.py`.

Hit@k and MRR are the gate metrics. Precision@4 stays low when a query has one gold listing and the retriever returns four chunks — that is expected, not a quality claim. These numbers are a regression check on a dated sample corpus, not live inventory and not a published IR benchmark. Refreshing `data/seattle_rentals.jsonl` can invalidate gold ids; update the gold file and re-run the gate after a snapshot change.

## Roadmap

Retrieval evaluation is in place (gold queries, ranking metrics, and a regression gate). Next is generation-side eval over the same snapshot, still without treating this as live inventory.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
