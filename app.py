import streamlit as st
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


st.set_page_config(page_title="LeaseGPT",page_icon=':shark:')

def main():
    st.title("LeaseGPT")
    st.write("Your AI Leasing Assistant")
    selection = st.selectbox("Choose a city", ["Seattle", "LA", "San Francisco", "New York City"])
    if selection == "Seattle":
        st.write("You selected Seattle.")
    elif selection == "LA":
        st.write("You selected LA.")
    elif selection == "San Francisco":
        st.write("You selected San Francisco.")
    elif selection == "New York City":
        st.write("You selected New York City.")
    else:
        st.write("You haven't selected anything yet.")

    st.sidebar.title("Hello ")
    st.sidebar.write("This is your personal leasing agent")


if __name__ == '__main__':
    main()

