import os
import pickle

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from leasegpt.listings import doc1, doc2, doc3, doc4


def get_text_chunks(selection: str):
    # TODO: Scraping Craigslist
    text = " ".join([doc1, doc2, doc3, doc4])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=50, length_function=len
    )
    docs = [doc1, doc2, doc3, doc4]
    chunks = []

    # Splitting the text into chunks
    for doc in docs:
        if len(doc) > 1200:
            chunk_doc = text_splitter.split_text(doc)
            for chunk in chunk_doc:
                chunks.append(chunk)
        else:
            chunks.append(doc)
    return chunks


def get_set_vector_store(chunks, selection):
    embeddings = OpenAIEmbeddings()
    # Local demo cache only — not portable or safe to share.
    store_name = "craigslist_vector_store"
    if os.path.exists(f"{store_name}.pkl"):
        with open(f"{store_name}.pkl", "rb") as f:
            vector_store = pickle.load(f)
            st.write("Embeddings Loaded from the Disk")
    else:
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"{store_name}.pkl", "wb") as f:
            pickle.dump(vector_store, f)
            st.write("Embeddings Created and Saved to Disk")
    return vector_store
