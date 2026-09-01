import os

import streamlit as st
from dotenv import load_dotenv
from groq import APIError, AuthenticationError

from leasegpt.generator import generate_response, setup_leasing_agent
from leasegpt.retriever import get_set_vector_store, get_text_chunks, retrieve_sources
from leasegpt.ui import (
    apply_styles,
    render_chat,
    render_composer,
    render_empty_state,
    render_header,
    render_listings_tab,
    render_sidebar,
    render_sources_panel,
)

st.set_page_config(page_title="LeaseGPT", page_icon=":door:", layout="centered")


def _resolve_api_key(sidebar_key: str) -> str:
    key = sidebar_key or os.environ.get("GROQ_API_KEY", "") or ""
    if key:
        os.environ["GROQ_API_KEY"] = key
    return key


def _get_cached_rag(api_key: str, selection: str):
    if st.session_state.get("vector_store") is None:
        chunks = get_text_chunks(selection)
        st.session_state.vector_store = get_set_vector_store(chunks, selection)
    key_changed = st.session_state.get("agent_api_key") != api_key
    if api_key and (st.session_state.get("leasing_agent") is None or key_changed):
        st.session_state.leasing_agent = setup_leasing_agent(
            st.session_state.vector_store, api_key
        )
        st.session_state.agent_api_key = api_key
    return st.session_state.vector_store, st.session_state.get("leasing_agent")


def _init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def _run_turn(query: str, api_key: str):
    vector_store, agent = _get_cached_rag(api_key, "Seattle")
    sources = retrieve_sources(vector_store, query)
    answer = generate_response(agent, query)
    st.session_state.messages.append({"role": "user", "content": query, "sources": []})
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main():
    load_dotenv()
    _init_state()
    apply_styles()

    env_key = os.environ.get("GROQ_API_KEY", "") or ""
    sidebar = render_sidebar(api_key_present=bool(env_key))
    api_key = _resolve_api_key(sidebar["sidebar_key"])

    render_header()

    query = st.session_state.pop("pending_query", None)
    if not api_key:
        st.info("Enter a Groq API key in the sidebar to chat, or set GROQ_API_KEY.")

    if query:
        if not api_key:
            st.warning("Add an API key before running a query.")
        else:
            try:
                with st.spinner("Searching listings and drafting an answer…"):
                    _run_turn(query, api_key)
            except (KeyError, AuthenticationError, APIError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(str(exc))

    messages = st.session_state.messages
    latest_sources = []
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg.get("sources"):
            latest_sources = msg["sources"]
            break

    chat_tab, sources_tab, listings_tab = st.tabs(["Chat", "Sources", "Listings"])
    with chat_tab:
        if not messages:
            render_empty_state()
        else:
            render_chat(messages)
        composed = render_composer()
        if composed:
            st.session_state.pending_query = composed
            st.experimental_rerun()
    with sources_tab:
        render_sources_panel(latest_sources)
    with listings_tab:
        render_listings_tab()


if __name__ == "__main__":
    main()
