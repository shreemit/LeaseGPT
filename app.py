import streamlit as st
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
import pickle
import os
from streamlit_chat import message
from dotenv import load_dotenv
from raw_strings import *
import openai

st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":door:")

def get_listings_tool(retriever):
    tool_desc = '''Use this tool to answer user questions using Apartment listings from Craigslist. If the user asks a question that is not in the listings, the tool will use OpenAI to generate a response.
    This tool can also be used for follow up quesitons from the user. 
    '''
    tool = Tool(
        func=retriever,
        description=tool_desc,
        name="Lease Listings Tool",   
    )
    return tool

def main():
    os.environ["OPENAI_API_KEY"] = ""
    st.title("🚪🏡 LeaseGPT")
    st.write("Your AI Leasing Assistant")

    print("Key", os.environ["OPENAI_API_KEY"])
    
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

        # Splitting the text into chunks
        for doc in docs:
            if len(doc) > 1200:
                chunk_doc = text_splitter.split_text(doc)
                for chunk in chunk_doc:
                    chunks.append(chunk)
            else:
                chunks.append(doc)

        # st.write("Number of chunks", chunks)
        # chunks = text_splitter.split_text(text)

        store_name = "craigslist"
        template =  '''I want you to act to act like a leasing agent for me. Giving me the best options always based on what you read below. 
        You can give me something which matches my criteria or something which is close to it.
        Question: {question}
        Answer:
        '''
        prompt = PromptTemplate(template=template, input_variables=["question"])


        # print("Prompt", prompt)
        try:
            embeddings = OpenAIEmbeddings()
            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
                    st.write("Embeddings Loaded from the Disk")
            else:
                
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)
                    st.write("Embeddings Created and Saved to Disk")

            query = st.text_input("Ask your question")
            
            if query:
                docs = VectorStore.similarity_search(query, k=3)
                st.write("Docs", docs)
                llm = OpenAI(model_name="gpt-3.5-turbo")
                retriever = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriver=VectorStore.as_retriever()
                )

                tools = [get_listings_tool(retriever=retriever)]
                memory = ConversationBufferWindowMemory(
                    memory_key='chat_history',
                    k=3,
                    return_messages=True
                )

                conversational_agent = initialize_agent(
                    agent='chat-conversational-react-description',
                    tools=tools,
                    llm=llm,
                    verbose=True,
                    max_iterations=2,
                    early_stopping_method='generate',
                    memory=memory
                ) 

                conversational_prompt = conversational_agent.agent.create_prompt(
                    system_message = template,
                    tools=tools,
                )

                st.write("Before", conversational_agent.agent.llm_chain.prompt)

                conversational_agent.agent.llm_chain.prompt = conversational_prompt

                st.write("After", conversational_agent.agent.llm_chain.prompt)

        
                print("agent", conversational_agent.agent.llm_chain.prompt)



                prompt.format(question=query)
                question = prompt.format(question=query)
                print("Question", question)
                # st.write("Prompt", question)
                # st.write("Docs", docs)
                with get_openai_callback() as callback:
                    # response = chain.run(input_documents=docs, question=question)
                    # print("Response", response)
                    # st.write(chain)
                    st.write("Cost for query", callback.total_cost)
                    
                #     st.write(response)
                # print("Response", response) 
        except openai.error.AuthenticationError as e:
            # print("Error", e)
            st.write("Please enter a valid OpenAI API Key")
        except:
            if os.environ["OPENAI_API_KEY"] is None:
                st.write("Please enter an OpenAI API Key")
            # if e == "No API key found":

    st.sidebar.title("Hello")
    st.sidebar.write("This is your personal leasing agent LeasingGPT")
    st.sidebar.write("I can help you find the best apartment for you")
    st.sidebar.write("Made by Shreemit https://github.com/shreemit")

if __name__ == "__main__":
    main()


