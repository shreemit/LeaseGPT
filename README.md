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

The checked-in snapshot is `data/seattle_rentals.jsonl` plus `data/seattle_rentals.meta.json` (`fetched_at` and query metadata). It is dated sample inventory, not live MLS. RentCast [API Terms](https://www.rentcast.io/terms-api) §1 allow storing, displaying, and distributing API Data; that snapshot is **not** covered by this repo’s MIT license (see `LICENSE`). Agent/office phones, emails, and websites are stripped before check-in. Do not share `RENTCAST_API_KEY`.

Refresh locally (50 free RentCast requests/month; do not call this from Streamlit). The script updates matching ids, appends new ones, and drops listings that disappeared only when it finishes a full API page walk:

```sh
RENTCAST_API_KEY=... uv run python scripts/fetch_rentcast_listings.py
uv run python scripts/build_vector_store.py
```

The vector index is local FastEmbed + FAISS. After a listing refresh, rebuild it with the second command (or let the app build on first chat). The index under `data/faiss/` is gitignored.

Firefox/geckodriver is only required if you run `leasegpt/scraper.py` yourself.

## Tests

No extra test runner package. From the repo root, after `uv sync`:

```sh
uv run python -m unittest discover -s tests -t . -v
```

That suite is offline: it loads the checked-in Seattle snapshot, filters listings, checks retriever chunking / `retrieve_sources` on a tiny in-memory index, and statically asserts app modules do not import `leasegpt.scraper` or `scripts/fetch_rentcast_listings.py`. It does not need Groq, RentCast, or network.

FastEmbed/ONNX embedding checks are gated. The default command skips them so CI stays fast. To run the optional embed:

```sh
LEASEGPT_TEST_FASTEMBED=1 uv run python -m unittest tests.test_retriever.FastEmbedGatedTests -v
```

This is unit coverage of listing/retriever helpers only. For retrieval ranking and faithfulness reports, see [Evaluation](#evaluation).

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

## Evaluation

LeaseGPT includes a repeatable evaluation layer over 25 snapshot-grounded
queries. It reports retrieval precision/recall at 3, 5, and 10 for the current
dense retriever and an eval-only constraint re-ranker. With `GROQ_API_KEY`, it
also uses a separate rubric-driven LLM judge to score answer faithfulness and
relevance.

```sh
uv run python -m eval.run_eval
uv run python -m eval.run_eval --skip-llm
```

See the checked-in methodology and findings in
[`eval/results.md`](eval/results.md), or open the self-contained
[`eval/results.html`](eval/results.html) dashboard directly in a browser. Both
are generated from the same `eval/results.json` artifact; viewing them does not
run evaluation or modify the app's index.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
