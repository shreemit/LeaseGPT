# LeaseGPT — agent notes

RAG leasing demo: frozen Seattle RentCast snapshot (`data/seattle_rentals.jsonl`) → chunk + FastEmbed + in-memory FAISS → LangChain conversational agent (Groq) → Streamlit tabs. Keep changes small. This is not live inventory or a production leasing platform. The app must not call RentCast.

Recent PRs that define the current shape: [#1](https://github.com/shreemit/LeaseGPT/pull/1) (package + uv), [#2](https://github.com/shreemit/LeaseGPT/pull/2) (tabbed UI, FastEmbed, Groq, hosting files). Do not resurrect deleted POCs (`chatPOC.py`, `llama_POC.py`, `llmTest.py`) or route chat through OpenAI / `langchain-openai`.

## Commands

Python **3.10** is required (`faiss-cpu==1.7.3` has no 3.13 wheels). Use [uv](https://docs.astral.sh/uv/) locally.

```sh
uv sync
uv run streamlit run app.py
uv run python -m unittest discover -s tests -t . -v
uv run python -m eval.run_eval --skip-llm
```

- Chat needs a Groq key ([console.groq.com](https://console.groq.com)): sidebar field, or `GROQ_API_KEY` in a gitignored `.env`. Hosted deploys use a Space / Community Cloud secret.
- Retrieval and the Listings tab work without a Groq key. Do not rebuild the vector store behind a Groq/OpenAI/RentCast key.
- Refresh listings only via `uv run python scripts/fetch_rentcast_listings.py` with `RENTCAST_API_KEY` in the environment or a gitignored `.env` at the repo root or `leasegpt/.env`. Never import that script from the app.
- Firefox/geckodriver: only if you run `leasegpt/scraper.py`. The app must not import the scraper (it launches Firefox at import time).

There is no dedicated test runner package. Offline unit tests (snapshot listings, retriever helpers, import guards, eval metrics) use stdlib `unittest` and need no API keys:

```sh
uv run python -m unittest discover -s tests -t . -v
```

Listing/filter/import-guard tests always run. Retriever tests use a stub or tiny hash-embedding FAISS index by default. FastEmbed/ONNX checks are skipped unless `LEASEGPT_TEST_FASTEMBED=1` (first run may download `BAAI/bge-small-en-v1.5`). Ranking and faithfulness reports live under `eval/` (`uv run python -m eval.run_eval`); do not treat the unit suite as a substitute for that harness.

## Layout

| Path | Role |
|------|------|
| `app.py` | Streamlit entry: session state, key resolution, RAG cache, tabs, chat turn |
| `leasegpt/listings.py` | Loads `data/seattle_rentals.jsonl` + `.meta.json` → `Listing` / `SAMPLE_LISTINGS` / `filter_listings` |
| `data/seattle_rentals.jsonl` | Frozen RentCast Active Seattle snapshot (one listing per line) |
| `data/seattle_rentals.meta.json` | Snapshot `fetched_at` / query metadata |
| `scripts/fetch_rentcast_listings.py` | Offline RentCast pull; **do not import from the app** |
| `scripts/build_vector_store.py` | Offline FAISS build into gitignored `data/faiss/`; **do not import from the app** |
| `leasegpt/retriever.py` | Chunking, FastEmbed (`BAAI/bge-small-en-v1.5`) + in-memory FAISS, `retrieve_sources` |
| `leasegpt/generator.py` | RetrievalQA tool + `chat-conversational-react-description` agent |
| `leasegpt/groq_chat.py` | LangChain 0.0.181 `SimpleChatModel` over the Groq SDK (`openai/gpt-oss-20b`) |
| `leasegpt/ui.py` | CSS, sidebar (key only), Chat empty/composer, Sources panel, Listings filters |
| `leasegpt/scraper.py` | Standalone Selenium Craigslist script; **do not import from the app** |
| `eval/` | Offline RAG dataset, retrieval metrics/config comparison, LLM judge, Markdown report, and static HTML dashboard |
| `tests/` | Offline `unittest` coverage for listings, retriever helpers, and import guards |
| `Dockerfile` / `requirements.txt` | Hugging Face Spaces (Docker) and Streamlit Community Cloud |
| `pyproject.toml` / `uv.lock` | Local uv pins — keep `requirements.txt` in sync when deps change |

UI: `layout="centered"`, tabs **Chat / Sources / Listings**. Sidebar is the Groq key only. Price/neighborhood filters live on **Listings** and affect sample cards only; retrieval always searches the full index. Chat expander (“Why this answer”) lists listing titles; Sources shows retrieved chunks.

## Conventions

- Chat: `leasegpt.groq_chat.ChatGroq` only. Do not use `ChatOpenAI` / `langchain-openai`; those are incompatible with LangChain `0.0.181` `LLMChain` after PR #2.
- Embeddings: local FastEmbed ONNX. Process-level cache in `retriever._vector_store`. Optional gitignored FAISS at `data/faiss/` (built with `uv run python scripts/build_vector_store.py`); the app loads it when the stamp matches the snapshot, otherwise it builds on first use. Do not check in the index.
- `retrieve_sources` is display-only. Do not fold it into the generation chain unless the user asks to change grounding.
- **Never write a pasted API key into `os.environ`.** Hosted Streamlit shares one process; `_resolve_api_key` must stay session-scoped (sidebar key first, then env secret).
- Example-query buttons in `render_empty_state` must **return** the chosen string. `app.py` queues `pending_query` and calls `st.experimental_rerun()`. Setting session state only inside the button does not run a turn (fixed in PR #2).
- Streamlit is **1.19.0**: `st.experimental_rerun()` is correct; do not switch to `st.rerun()` without bumping Streamlit.
- Pin LangChain at `0.0.181`. Casual upgrades break `initialize_agent` / `RetrievalQA`.
- Escape listing text before `unsafe_allow_html`.
- New sample listings: refresh `data/seattle_rentals.jsonl` with the fetch script (or add records there). Do not hardcode Craigslist ads back into `listings.py`.
- Hosting: GitHub Pages cannot run this. Spaces use Docker (`Dockerfile` bakes FastEmbed weights). Community Cloud installs from `requirements.txt`. README YAML frontmatter is for Hugging Face.

## Do not

- Scrape Craigslist or other sites from the app, CI, or “better retrieval” unless the user explicitly asks for scraper work.
- Call RentCast (or any listings API) from Streamlit, hosted deploys, or chat turns.
- Broaden city/corpus or claim live inventory; answers stay grounded in the checked-in snapshot.
- Check in `.env` or vector-store pickles.
- Mass-upgrade to “modern LangChain” without a dedicated migration.

## UI / deploy checks

Verify Chat empty state + example buttons **and** the Ask form (both must produce a turn with a key). Without a key: no crash, Listings still works, chat warns. After an answer: Sources has chunks; Chat expander has titles. Listings filters change cards only. If you change deps, update both `pyproject.toml` and `requirements.txt`.
