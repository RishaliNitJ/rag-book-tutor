# 📚 RAG Book Tutor

An LLM-powered **Retrieval-Augmented Generation (RAG)** system that allows users to query and understand PDF books interactively.

---

## 🚀 Overview

RAG Book Tutor enables users to upload PDF documents and ask questions about their content.
The system retrieves relevant information from the document and generates accurate answers using a Large Language Model (LLM).

---

## ✨ Features

* 📄 Upload and process PDF books
* 🔍 Intelligent text chunking and embedding
* ⚡ Fast semantic search using vector database (FAISS)
* 🤖 LLM-powered answer generation (Groq / HuggingFace)
* 🧠 Context-aware responses based on document content

---

## 🛠️ Tech Stack

* **Language:** Python
* **Frameworks/Libraries:**

  * LangChain
  * FAISS (Vector Store)
  * HuggingFace Embeddings
  * Groq API / LLM
* **Other Tools:**

  * PyPDF (PDF parsing)
  * Streamlit (UI, if used)

---

## ⚙️ How It Works

1. 📥 Load PDF document
2. ✂️ Split text into smaller chunks
3. 🔢 Convert text into embeddings
4. 🗂️ Store embeddings in FAISS vector database
5. ❓ User asks a query
6. 🔍 Retrieve relevant chunks
7. 🤖 Generate answer using LLM

---

## 📦 Installation

```bash
git clone https://github.com/RishaliNitJ/rag-book-tutor.git
cd rag-book-tutor
pip install -r requirements.txt
```

---

## 📁 Project Structure

```
rag-book-tutor/
│── app.py                # Main application
│── file.py               # Helper functions
│── requirements.txt      # Dependencies
│── test_groq_api.py      # API testing
│── test_hf_api.py        # HuggingFace testing
│── README.md
```

---

## 📌 Example Use Cases

* 📖 Study assistant for textbooks
* 📊 Research paper summarization
* 🧑‍🎓 Exam preparation tool
* 📚 Quick knowledge retrieval from large PDFs

---

## 🌟 Future Improvements

* Add chat history memory
* Support multiple PDFs
* Improve UI with Streamlit
* Deploy as a web app

---

## 👩‍💻 Author

**Rishali **

---

## ⭐ If you like this project

Give it a star on GitHub ⭐
