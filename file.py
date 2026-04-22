from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub

# ---------- UI ----------
st.set_page_config(page_title="Book Tutor RAG (HF API)", layout="wide")
st.title("📚 RAG Book Tutor - HuggingFace API")

uploaded_file = st.file_uploader("Upload a PDF Book", type=["pdf"])
query = st.text_input("Ask something")
mode = st.selectbox(
    "Mode",
    ["Long Summary", "Question Generation", "MCQ Generator"]
)

# ---------- Functions ----------

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


def build_vector_db(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)

    embeddings = HuggingFaceInferenceAPIEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    db = FAISS.from_texts(chunks, embeddings)
    return db


def retrieve_context(db, query, k=6):
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([d.page_content for d in docs])


def run_llm(context, mode):
    llm = HuggingFaceHub(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        model_kwargs={"temperature":0.2, "max_new_tokens":800}
    )

    if mode == "Long Summary":
        prompt = f"""
You are an academic summarizer.
Summarize ONLY using the provided context.
Do NOT use any external knowledge.
If something is missing, say "Not found in the book".

Context:
{context}

Generate a detailed structured summary.
"""
    elif mode == "Question Generation":
        prompt = f"""
You are a question generator.
Based ONLY on the context generate:
- 5 factual questions
- 5 conceptual questions
- 5 analytical questions

Context:
{context}
"""
    else:
        prompt = f"""
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

    return llm.invoke(prompt)

# ---------- Main ----------

if uploaded_file:
    with st.spinner("Indexing book..."):
        book_text = load_pdf(uploaded_file)
        db = build_vector_db(book_text)
        st.success("Book indexed successfully!")

    if query:
        with st.spinner("Generating..."):
            context = retrieve_context(db, query)
            answer = run_llm(context, mode)

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Retrieved Context"):
            st.write(context)
