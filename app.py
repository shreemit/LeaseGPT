import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import pickle
import os
from streamlit_chat import message
from dotenv import load_dotenv
from raw_strings import *

st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":shark:")


def main():
    st.title("🚪🏡 LeaseGPT")
    st.write("Your AI Leasing Assistant")
    
    selection = st.selectbox(
        "Choose a city", ["Seattle", "LA", "San Francisco", "New York City"]
    )
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

    api_key = st.text_input("Please enter your OpenAI key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    load_dotenv()

    if os.environ["OPENAI_API_KEY"] is not None:
        print("OPEN AI Key", os.environ["OPENAI_API_KEY"])
        # TODO: Scraping Craigslist

        # Combinig all the text into one string
        text = " ".join([doc1, doc2, doc3, doc4])

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=50, length_function=len
        )

        docs = [doc1, doc2, doc3, doc4]

        chunks = []

        # # Splitting the text into chunks
        for doc in docs:
            if len(doc) > 1000:
                chunk_doc = text_splitter.split_text(doc)
                for chunk in chunk_doc:
                    chunks.append(chunk)
            else:
                chunks.append(doc)

        store_name = "craigslist"
        template =  '''I want you to act to act like a leasing agent for me. Giving me the best options always based on what you read below. 
        You can give me something which matches my criteria or something which is close to it.
        Question: {question}
        Answer:
        '''
        prompt = PromptTemplate(template=template, input_variables=["question"])
        # print("Prompt", prompt)
        try:
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                    # st.write("Embeddings Loaded from the Disk")
            else:
                embeddings = OpenAIEmbeddings()
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    # st.write("Embeddings Created and Saved to Disk")

            query = st.text_input("Ask your question")
            if query:
                docs = VectorStore.similarity_search(query, k=3)
                davinci = OpenAI(model_name="text-davinci-003")
                chain = load_qa_chain(llm=davinci, chain_type="stuff")
                prompt.format(question=query)
                question = prompt.format(question=query)
                # st.write("Prompt", question)
                # st.write("Docs", docs)
                with get_openai_callback() as callback:
                    response = chain.run(input_documents=docs, question=template)
                    # st.write(chain)
                    st.write("Cost for query", callback.total_cost)
                    st.write(response)
                print("Response", response)

        except:
            if os.environ["OPENAI_API_KEY"] is None:
                st.write("Please enter a valid OpenAI API Key")

    st.sidebar.title("Hello ")
    st.sidebar.write("This is your personal leasing agent LeasingGPT")
    st.sidebar.write("I can help you find the best apartment for you")
    st.sidebar.write("Made by Shreemit")

if __name__ == "__main__":
    main()
