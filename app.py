import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
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

    template = """Question: {question}

    Answer: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    davinci = OpenAI(model_name='text-davinci-003')

    llm_chain = LLMChain(
        prompt=prompt,
        llm=davinci
    )

    qs = [
        {'question': "Which NFL team won the Super Bowl in the 2010 season?"},
        {'question': "If I am 6 ft 4 inches, how tall am I in centimeters?"},
        {'question': "Who was the 12th person on the moon?"},
        {'question': "How many eyes does a blade of grass have?"}
    ]
    llm_chain.generate(qs)

    st.sidebar.title("Hello ")
    st.sidebar.write("This is your personal leasing agent")


    



if __name__ == '__main__':
    main()

