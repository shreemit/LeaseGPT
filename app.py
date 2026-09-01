import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from dotenv import load_dotenv
import os

from leasegpt.retriever import get_text_chunks, get_set_vector_store
from leasegpt.generator import setup_leasing_agent, generate_response

st.set_page_config(page_title="🏡 LeaseGPT", page_icon=":door:")
# st.session_state.input = ""


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
    colored_header(label="", description="", color_name= "blue-40")

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

            # except openai.error.AuthenticationError as e:
            #     # print("Error", e)
            #     st.write("Please enter a valid OpenAI API Key")
            except:
                if os.environ["OPENAI_API_KEY"] is None:
                    st.write("Please enter an OpenAI API Key")

       # Check if there are any generated messages stored in the Streamlit session state
        if st.session_state["generated"]:
            # Loop through each generated message
            for i in range(len(st.session_state["generated"])):
                # Display the user message in the chat interface
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                # Display the generated message in the chat interface
                message(st.session_state["generated"][i], key=str(i))



if __name__ == "__main__":
    main()

# Sidebar contents
