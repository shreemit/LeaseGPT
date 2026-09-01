import os

import openai
import streamlit as st
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header

from leasegpt.generator import generate_response, setup_leasing_agent
from leasegpt.retriever import get_set_vector_store, get_text_chunks


def add_vertical_space(num_lines: int = 1):
    for _ in range(num_lines):
        st.write("")


st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":door:")


def clear_text():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ""


def _resolve_api_key(sidebar_key: str) -> str:
    if sidebar_key:
        os.environ["OPENAI_API_KEY"] = sidebar_key
        return sidebar_key
    return os.environ.get("OPENAI_API_KEY", "") or ""


def _get_cached_agent(api_key: str, selection: str):
    if (
        st.session_state.get("leasing_agent") is None
        or st.session_state.get("agent_api_key") != api_key
    ):
        chunks = get_text_chunks(selection)
        vector_store = get_set_vector_store(chunks, selection)
        st.session_state.leasing_agent = setup_leasing_agent(vector_store, api_key)
        st.session_state.agent_api_key = api_key
    return st.session_state.leasing_agent


def main():
    load_dotenv()

    with st.sidebar:
        st.markdown(
            """
        # Hello 👋
        ### This is your personal leasing agent LeaseGPT
        ### I can help you find the best apartment for you
        """
        )

        add_vertical_space(3)
        selection = st.selectbox(
            "Choose your city",
            ["Seattle", "LA", "San Francisco", "New York City"],
            disabled=True,
        )
        st.caption(
            "Listings are a hardcoded Seattle sample. City filtering is not wired yet."
        )
        sidebar_key = st.text_input("Please enter your OpenAI key", type="password")
        api_key = _resolve_api_key(sidebar_key)

        add_vertical_space(15)
        st.markdown("Made by Shreemit [Github](https://github.com/shreemit/LeaseGPT)")

    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I'm LeaseGPT, How may I help you?"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hi!"]
    if "something" not in st.session_state:
        st.session_state.something = ""

    st.title("🚪🏡 LeaseGPT")
    st.write("Your AI Leasing Assistant")
    colored_header(label="", description="", color_name="blue-30")
    response_container = st.container()
    input_container = st.container()
    colored_header(label="", description="", color_name="blue-40")

    with input_container:
        user_input = st.text_input("User: ", key="widget")
        print("User Input 1", user_input)

    if not api_key:
        st.info("Enter an OpenAI API key in the sidebar to chat.")

    with response_container:
        if user_input and api_key:
            try:
                leasing_gpt = _get_cached_agent(api_key, selection)
                print("User Input", user_input)
                response = generate_response(leasing_gpt, user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)
            except (KeyError, openai.AuthenticationError, openai.APIError) as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(str(exc))

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))


if __name__ == "__main__":
    main()
