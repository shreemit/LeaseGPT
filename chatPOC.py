# import streamlit as st
# from streamlit_chat import message
# from streamlit_extras.colored_header import colored_header
# from streamlit_extras.add_vertical_space import add_vertical_space
# from hugchat import hugchat

# st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

# with st.sidebar:
#     st.title('🤗💬 HugChat App')
#     st.markdown('''
#     ## About
#     This app is an LLM-powered chatbot built using:
#     - [Streamlit](<https://streamlit.io/>)
#     - [HugChat](<https://github.com/Soulter/hugging-chat-api>)
#     - [OpenAssistant/oasst-sft-6-llama-30b-xor](<https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor>) LLM model
#     ''')
#     add_vertical_space(5)

# response_container = st.container()
# colored_header(label='', description='', color_name='blue-30')
# input_container = st.container()

# ## Function for taking user provided prompt as input
# def get_text():
#     input_text = st.text_input("You: ", "", key="input_text", on_change=generate_answer)
#     return input_text

# ## Applying the user input box
# with input_container:
#     user_input = get_text()

# with response_container:
#     if user_input:
#         response = "Yes I can help you with that."
#         st.session_state.past.append(user_input)
#         st.session_state.generated.append(response)
        
#     if st.session_state['generated']:
#         for i in range(len(st.session_state['generated'])):
#             message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
#             message(st.session_state['generated'][i], key=str(i))
#             st.session_state['input_text'] = ""



# if "history" not in st.session_state:
#     st.session_state.history = []

# st.title("Hello Chatbot")


# def generate_answer():
#     user_message = st.session_state.input_text

#     st.session_state.history.append({"message": user_message, "is_user": True})
#     st.session_state.history.append({"message": "message_bot", "is_user": False})
#     st.session_state.input_text = ""



# for i, chat in enumerate(st.session_state.history):
#     st_message(**chat, key=str(i)) #unpacking

# # st.text_input("Talk to the bot", key="input_text", on_change=generate_answer)


import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")

# Sidebar contents
with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [HugChat](https://github.com/Soulter/hugging-chat-api)
    - [OpenAssistant/oasst-sft-6-llama-30b-xor](https://huggingface.co/OpenAssistant/oasst-sft-6-llama-30b-xor) LLM model
    
    💡 Note: No API key required!
    ''')
    add_vertical_space(5)

# Generate empty lists for generated and past.
if 'generated' not in st.session_state:
    st.session_state['generated'] = ["I'm HugChat, How may I help you?"]
## past stores User's questions
if 'past' not in st.session_state:
    st.session_state['past'] = ['Hi!']
    
# Layout of input/response containers
input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text
## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):

    return "Yes I can help you with that."

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
