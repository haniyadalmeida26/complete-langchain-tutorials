import streamlit as st
import os

from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")


# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Chat with Multiple PDFs",
    page_icon="📚",
    layout="wide"
)

st.header("📚 Chat with Multiple PDFs using Cohere")


# ============================================================
# CHECK API KEY
# ============================================================

if not COHERE_API_KEY:

    st.error(
        "COHERE_API_KEY is missing. "
        "Please create a .env file and add your Cohere API key."
    )

    st.stop()


# ============================================================
# COHERE MODELS
# ============================================================

chat_model = ChatCohere(
    model="command-a-03-2025",
    temperature=0.3,
    cohere_api_key=COHERE_API_KEY
)


embeddings = CohereEmbeddings(
    model="embed-v4.0",
    cohere_api_key=COHERE_API_KEY
)


# ============================================================
# EXTRACT TEXT FROM PDFS
# ============================================================

def get_pdf_text(pdf_files):

    text = ""

    for pdf in pdf_files:

        pdf_reader = PdfReader(pdf)

        for page in pdf_reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ============================================================
# SPLIT TEXT INTO CHUNKS
# ============================================================

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=500
    )

    chunks = text_splitter.split_text(text)

    return chunks


# ============================================================
# CREATE FAISS VECTOR STORE
# ============================================================

def get_vector_store(text_chunks):

    vector_store = FAISS.from_texts(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# ============================================================
# GET ANSWER FROM COHERE
# ============================================================

def get_answer(user_question):

    # Load previously created FAISS database
    new_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    # Find relevant document chunks
    docs = new_db.similarity_search(
        user_question,
        k=4
    )

    # Combine relevant chunks
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    # Prompt for Cohere
    prompt = f"""
You are a helpful PDF question-answering assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not present in the context, say:

"Answer is not available in the provided PDF documents."

Do not make up information.

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{user_question}

---------------- ANSWER ----------------
"""

    # Get response
    response = chat_model.invoke(prompt)

    return response.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📚 PDF Menu")

    pdf_files = st.file_uploader(
        "Upload your PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    process_button = st.button(
        "Submit & Process"
    )


# ============================================================
# PROCESS PDFS
# ============================================================

if process_button:

    if not pdf_files:

        st.warning(
            "⚠️ Please upload at least one PDF file."
        )

    else:

        with st.spinner(
            "Reading and processing your PDFs... 📖"
        ):

            try:

                # Extract text
                raw_text = get_pdf_text(pdf_files)

                if not raw_text.strip():

                    st.error(
                        "Could not extract text from the uploaded PDFs."
                    )

                else:

                    # Split into chunks
                    text_chunks = get_text_chunks(
                        raw_text
                    )

                    # Create vector store
                    get_vector_store(
                        text_chunks
                    )

                    st.success(
                        f"Done! Processed {len(pdf_files)} PDF(s) "
                        f"and created {len(text_chunks)} text chunks."
                    )

            except Exception as e:

                st.error(
                    f"Error while processing PDFs:\n\n{e}"
                )


# ============================================================
# USER QUESTION
# ============================================================

user_question = st.text_input(
    "Ask a question from your PDF files:",
    placeholder="Example: What is the main topic discussed in the document?"
)


# ============================================================
# ANSWER QUESTION
# ============================================================

if user_question:

    if not os.path.exists("faiss_index"):

        st.warning(
            "⚠️ Please upload and process your PDF files first."
        )

    else:

        with st.spinner("Searching your PDFs... 🔍"):

            try:

                answer = get_answer(
                    user_question
                )

                st.subheader("💬 Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error while answering the question:\n\n{e}"
                )