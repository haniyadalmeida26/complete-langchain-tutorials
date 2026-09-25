import streamlit as st
import os

from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


if not COHERE_API_KEY:
    st.error("COHERE_API_KEY is not found. Please check your .env file.")
    st.stop()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.set_page_config(
    page_title="Conversational Q&A Chatbot",
    page_icon="💬"
)

st.header("Hey, Let's Chat 💬")


# --------------------------------------------------
# Initialize Cohere Chat Model
# --------------------------------------------------

chat = ChatCohere(
    model="command-a-03-2025",
    temperature=0.5,
    cohere_api_key=COHERE_API_KEY
)


# --------------------------------------------------
# Initialize conversation history
# --------------------------------------------------

if "flowmessages" not in st.session_state:

    st.session_state["flowmessages"] = [
        SystemMessage(
            content="You are a comedian AI assistant."
        )
    ]


# --------------------------------------------------
# Function to get response
# --------------------------------------------------

def get_chatmodel_response(question):

    # Add user's question
    st.session_state["flowmessages"].append(
        HumanMessage(content=question)
    )

    # Get response from Cohere
    answer = chat.invoke(
        st.session_state["flowmessages"]
    )

    # Add AI response to conversation history
    st.session_state["flowmessages"].append(
        AIMessage(content=answer.content)
    )

    return answer.content


# --------------------------------------------------
# User Input
# --------------------------------------------------

user_input = st.text_input(
    "Input:",
    key="input"
)


# --------------------------------------------------
# Ask Button
# --------------------------------------------------

submit = st.button("Ask the question")


# --------------------------------------------------
# Generate response
# --------------------------------------------------

if submit:

    if user_input.strip():

        response = get_chatmodel_response(user_input)

        st.subheader("The Response is")

        st.write(response)

    else:

        st.warning("Please enter a question.")