import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pickle

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

    api_key = st.text_input('Please enter your OpenAI key')


    st.sidebar.title("Hello ")
    st.sidebar.write("This is your personal leasing agent")


    



if __name__ == '__main__':
    main()

