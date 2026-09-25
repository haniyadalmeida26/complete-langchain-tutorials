import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()


# Function to get response from Cohere model
def get_cohere_response(input_text, no_words, blog_style):

    # Cohere model
    llm = ChatCohere(
        model="command-a-03-2025",
        temperature=0.5,
        cohere_api_key=os.getenv("CO_API_KEY")
    )

    # Prompt Template
    template = """
    Write a blog for a {blog_style} audience about the topic {input_text}.
    Write the blog within {no_words} words.
    """

    prompt = PromptTemplate(
        input_variables=["blog_style", "input_text", "no_words"],
        template=template
    )

    # Generate response
    formatted_prompt = prompt.format(
        blog_style=blog_style,
        input_text=input_text,
        no_words=no_words
    )

    response = llm.invoke(formatted_prompt)

    return response.content


# Initialize Streamlit app
st.set_page_config(
    page_title="Generate Blogs",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two columns
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("No of Words")

with col2:
    blog_style = st.selectbox(
        "Writing the blog for",
        ("Researchers", "Data Scientist", "Common People"),
        index=0
    )

submit = st.button("Generate")


# Final response
if submit:

    if input_text and no_words:

        response = get_cohere_response(
            input_text,
            no_words,
            blog_style
        )

        st.subheader("Generated Blog")
        st.write(response)

    else:
        st.warning("Please enter the blog topic and number of words.")