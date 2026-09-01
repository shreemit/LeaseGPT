from langchain.agents import Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from leasegpt.groq_chat import GROQ_MODEL, ChatGroq


def get_listings_tool(retriever):
    tool_desc = """Use this tool to inform user about listings from context. Give the user 2 options based on their criterion. If the user asks a question that is not in the listings, the tool will generate a response from retrieved listing text.
    This tool can also be used for follow up quesitons from the user. 
    """
    tool = Tool(
        func=retriever.run,
        description=tool_desc,
        name="Lease Listings Tool",
    )
    return tool


def setup_leasing_agent(vector_store, api_key):
    template = """I want you to act to act like a leasing agent for me. Giving me the best options based on what you read below. 
        You can give me something which matches my criteria or something which is close to it. Always list the names of the listings and any other details like price. If you have details on the rent always list that as well.
        """

    llm = ChatGroq(groq_api_key=api_key, temperature=0, model=GROQ_MODEL)

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


def generate_response(conversational_agent, user_input):
    response = conversational_agent.run(user_input)
    return response
