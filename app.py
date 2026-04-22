from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


# ---------------- UI ----------------

st.set_page_config(page_title="RAG Book Tutor", layout="wide")
st.title("📚 RAG Book Tutor (Groq + Llama 3.1)")

uploaded_file = st.file_uploader("Upload a PDF Book", type=["pdf"])
query = st.text_input("Ask something about the book")

mode = st.selectbox(
    "Mode",
    ["Long Summary", "Question Generation", "MCQ Generator"]
)


# ---------------- Functions ----------------

@st.cache_data
def load_pdf(file):
    reader = PdfReader(file)
    text = ""

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:  # ✅ FIX: avoid None
            text += extracted

    return text


@st.cache_resource
def build_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_text(text)

    # ✅ FIX: handle empty chunks
    if len(chunks) == 0:
        raise ValueError("No chunks created from PDF text")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)
    return db


def retrieve_context(db, query, k=6):
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])


def get_prompt(context, mode):
    if mode == "Long Summary":
        return f"""
You are an academic summarizer.
Summarize ONLY using the provided context.
Do NOT use external knowledge.
If something is missing, say "Not found in the book".

Context:
{context}

Generate a detailed structured summary.
"""

    elif mode == "Question Generation":
        return f"""
You are a question generator.
Based ONLY on the context generate:
- 5 factual questions
- 5 conceptual questions
- 5 analytical questions

Context:
{context}
"""

    else:
        return f"""
Generate 10 MCQ questions from the context.
Each question must have:
- 4 options
- 1 correct answer
- short explanation

Format:
Q:
A)
B)
C)
D)
Correct:
Explanation:

Context:
{context}
"""


def run_llm(prompt):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
        max_tokens=800
    )
    return llm.invoke(prompt)


# ---------------- Main Pipeline ----------------

if uploaded_file:
    with st.spinner("Reading and indexing book..."):
        book_text = load_pdf(uploaded_file)

        # ✅ FIX: check empty PDF
        if not book_text.strip():
            st.error("❌ No readable text found in PDF (maybe scanned PDF)")
            st.stop()

        try:
            db = build_vector_db(book_text)
            st.success("Book indexed successfully!")
        except Exception as e:
            st.error(f"❌ Error while building vector DB: {e}")
            st.stop()

    if query:
        with st.spinner("Generating response..."):
            context = retrieve_context(db, query)
            prompt = get_prompt(context, mode)
            answer = run_llm(prompt)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context (RAG Evidence)"):
            st.write(context)