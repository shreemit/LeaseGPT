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

st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":door:")
# st.session_state.input = ""

# Sidebar contents
with st.sidebar:
    st.markdown('''
    # Hello 👋
    ### This is your personal leasing agent LeaseGPT
    ### I can help you find the best apartment for you
    ''')

    add_vertical_space(3)
    selection = st.selectbox(
        "Choose your city", ["Seattle", "LA", "San Francisco", "New York City"]
    )
    api_key = st.text_input("Please enter your OpenAI key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    load_dotenv()
    add_vertical_space(15)
    st.markdown('Made by Shreemit [Github](https://github.com/shreemit/LeaseGPT)')



# Generate empty lists for generated and past.
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm LeaseGPT, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
# Layout of input/response containers
if 'something' not in st.session_state:
    st.session_state.something = ''


st.title("🚪🏡 LeaseGPT")
st.write("Your AI Leasing Assistant")
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()
input_container = st.container()
colored_header(label='', description='', color_name='blue-40')

## Applying the user input box
with input_container:
    user_input = st.session_state.something

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    return "Yes I can help you with that."

def clear_text():
    st.session_state.something = st.session_state.widget
    st.session_state.widget = ''

st.text_input('User: ', key='widget', on_change=clear_text)

## Conditional display of AI generated responses as a function of user provided prompts
with response_container:
    if user_input:
        response = generate_response(user_input)
        st.session_state.past.append(user_input)
        st.session_state.generated.append(response)
        
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
    