import streamlit as st

from leasegpt.listings import (
    NEIGHBORHOODS,
    PRICE_MAX,
    PRICE_MIN,
    filter_listings,
)

EXAMPLE_QUERIES = [
    "3-bedroom near UW under $2800",
    "Something with in-unit laundry and parking",
    "Compare U-District options vs Ballard",
]

PAGE_CSS = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 760px;
    }
    .lease-kicker {
        color: #5c6570;
        font-size: 0.85rem;
        margin-top: -0.4rem;
        margin-bottom: 0.6rem;
    }
    .msg-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 0.2rem;
    }
    .source-card, .listing-card {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
        margin-bottom: 0.65rem;
        background: #fafafa;
    }
    .source-card h4, .listing-card h4 {
        margin: 0 0 0.25rem 0;
        font-size: 0.92rem;
        font-weight: 600;
    }
    .meta {
        color: #4b5563;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
    }
    .preview {
        color: #374151;
        font-size: 0.85rem;
        line-height: 1.4;
    }
</style>
"""


def apply_styles():
    st.markdown(PAGE_CSS, unsafe_allow_html=True)


def render_header():
    st.header("LeaseGPT")
    st.markdown(
        '<p class="lease-kicker">Ask in plain language; answers are grounded '
        "in a small Seattle listing sample.</p>",
        unsafe_allow_html=True,
    )


def render_sidebar(api_key_present: bool):
    with st.sidebar:
        st.caption("Seattle sample · RAG demo")
        sidebar_key = st.text_input("Groq API key", type="password")
        if api_key_present and not sidebar_key:
            st.caption("Using GROQ_API_KEY from the environment.")
        else:
            st.caption("[console.groq.com](https://console.groq.com)")
        st.caption("[GitHub](https://github.com/shreemit/LeaseGPT)")
        return {"sidebar_key": sidebar_key}


def render_empty_state():
    st.caption("Describe budget, neighborhood, or amenities — or try:")
    chosen = None
    for i, example in enumerate(EXAMPLE_QUERIES):
        if st.button(example, key=f"example_{i}"):
            chosen = example
    return chosen


def render_listings_tab():
    st.caption("Filters the sample cards only. Retrieval still uses the full index.")
    price_range = st.slider(
        "Price range",
        min_value=PRICE_MIN,
        max_value=PRICE_MAX,
        value=(PRICE_MIN, PRICE_MAX),
        format="$%d",
    )
    neighborhood = st.selectbox("Neighborhood", NEIGHBORHOODS)
    filtered = filter_listings(price_range[0], price_range[1], neighborhood)
    st.caption(f"{len(filtered)} of 4 listings · 3-bedroom sample · Seattle")
    for listing in filtered:
        _listing_card(listing)


def render_chat(messages):
    for i, msg in enumerate(messages):
        role = "You" if msg["role"] == "user" else "LeaseGPT"
        st.markdown(f'<div class="msg-label">{role}</div>', unsafe_allow_html=True)
        st.write(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Why this answer"):
                for source in msg["sources"]:
                    title = source.get("title") or "Retrieved chunk"
                    meta = _source_meta(source)
                    st.caption(f"{title}" + (f" · {meta}" if meta else ""))
        if i < len(messages) - 1:
            st.markdown("")


def render_sources_panel(sources):
    st.caption("Chunks retrieved for the latest answer. Open Chat for the generated reply.")
    if not sources:
        st.caption("Ask something in Chat to populate this tab.")
        return
    for source in sources:
        _source_card(source)


def render_composer():
    with st.form("ask_form", clear_on_submit=True):
        query = st.text_input(
            "Your question",
            placeholder="e.g. 3-bedroom near UW under $2800",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask")
    if submitted and query.strip():
        return query.strip()
    return None


def _listing_card(listing):
    st.markdown(
        f"""<div class="listing-card">
        <h4>{_escape(listing.title)}</h4>
        <div class="meta">${listing.cost:,} · { _escape(listing.neighborhood) } · { _escape(listing.bedrooms_label) }</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _source_card(source):
    meta = _source_meta(source)
    st.markdown(
        f"""<div class="source-card">
        <h4>{_escape(source.get("title") or "Retrieved chunk")}</h4>
        <div class="meta">{_escape(meta)}</div>
        <div class="preview">{_escape(source.get("preview") or "")}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _source_meta(source):
    parts = []
    if source.get("cost"):
        parts.append(f"${source['cost']:,}")
    if source.get("neighborhood"):
        parts.append(source["neighborhood"])
    return " · ".join(parts)


def _escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
