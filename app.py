import time
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat
from llmTest import get_listings_tool
from dotenv import load_dotenv
import os
from raw_strings import *
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.callbacks import get_openai_callback
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.chains import RetrievalQA
import openai

st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":door:")
# st.session_state.input = ""


def get_listings_tool(retriever):
    tool_desc = """Use this tool to inform user about listings from context. Give the user 2 options based on their criterion. If the user asks a question that is not in the listings, the tool will use OpenAI to generate a response.
    This tool can also be used for follow up quesitons from the user. 
    """
    tool = Tool(
        func=retriever.run,
        description=tool_desc,
        name="Lease Listings Tool",
    )
    return tool


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


def setup_leasing_agent(vector_store, api_key):
    template = """I want you to act to act like a leasing agent for me. Giving me the best options based on what you read below. 
        You can give me something which matches my criteria or something which is close to it. Always list the names of the listings and any other details. If you have details on the rent always list that as well.
        """

    llm = ChatOpenAI(openai_api_key=api_key, temperature=0, model_name="gpt-3.5-turbo")

    retriever = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )

    tools = [get_listings_tool(retriever=retriever)]
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", k=3, return_messages=True
    )

    conversational_agent = initialize_agent(
        agent="chat-conversational-react-description",
        tools=tools,
        llm=llm,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory,
    )

    conversational_prompt = conversational_agent.agent.create_prompt(
        system_message=template,
        tools=tools,
    )

    conversational_agent.agent.llm_chain.prompt = conversational_prompt
    print("Prompt", conversational_prompt)
    return conversational_agent


## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(conversational_agent, user_input):
    response = conversational_agent.run(user_input)
    return response


def clear_text():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ""


def main():
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
            "Choose your city", ["Seattle", "LA", "San Francisco", "New York City"]
        )
        api_key = st.text_input("Please enter your OpenAI key")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        load_dotenv()
        add_vertical_space(15)
        st.markdown("Made by Shreemit [Github](https://github.com/shreemit/LeaseGPT)")

    # Generate empty lists for generated and past.
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

    ## Applying the user input box
    with input_container:
        # user_input = st.session_state.widget
        user_input = st.text_input("User: ", key="widget")
        print("User Input 1", user_input)

    if os.environ["OPENAI_API_KEY"] != "":
        print("OPEN AI Key", os.environ["OPENAI_API_KEY"])
        chunks = get_text_chunks(selection)

    with response_container:
        if user_input:
            try:
                vectore_store = get_set_vector_store(chunks, selection)
                leasing_gpt = setup_leasing_agent(vectore_store, api_key)
                # with get_openai_callback() as callback:
                print("User Input", user_input)
                response = generate_response(leasing_gpt, user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(response)

            except openai.error.AuthenticationError as e:
                # print("Error", e)
                st.write("Please enter a valid OpenAI API Key")
            except:
                if os.environ["OPENAI_API_KEY"] is None:
                    st.write("Please enter an OpenAI API Key")

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))



if __name__ == "__main__":
    main()

# Sidebar contents
