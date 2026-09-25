import streamlit as st
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_cohere import ChatCohere


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Conversational Q&A Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.header("🤖 Hey, Let's Chat!")


# ============================================================
# CHECK API KEY
# ============================================================

if not COHERE_API_KEY:
    st.error(
        "COHERE_API_KEY is missing. "
        "Please check your .env file."
    )
    st.stop()


# ============================================================
# CREATE COHERE CHAT MODEL
# ============================================================

chat = ChatCohere(
    model="command-a-03-2025",
    temperature=0.5,
    cohere_api_key=COHERE_API_KEY
)


# ============================================================
# INITIALIZE CHAT HISTORY
# ============================================================

if "flowmessages" not in st.session_state:

    st.session_state["flowmessages"] = [
        SystemMessage(
            content=(
                "You are a comedian AI assistant. "
                "Answer the user's questions helpfully "
                "and add a little humor when appropriate."
            )
        )
    ]


# ============================================================
# FUNCTION TO GET RESPONSE
# ============================================================

def get_chatmodel_response(question):

    # Add user question to conversation history
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


# ============================================================
# USER INPUT
# ============================================================

user_input = st.text_input(
    "Input:",
    placeholder="Ask me anything..."
)


# ============================================================
# ASK BUTTON
# ============================================================

submit = st.button("Ask the question")


# ============================================================
# PROCESS QUESTION
# ============================================================

if submit:

    if user_input.strip():

        with st.spinner("Thinking... 🤔"):

            response = get_chatmodel_response(user_input)

        st.subheader("Response:")
        st.write(response)

    else:

        st.warning("⚠️ Please enter a question first.")