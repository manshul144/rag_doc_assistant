import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms.ollama import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings

# ---------------- CONFIG ----------------
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Embeddings + LLM
embeddings = OllamaEmbeddings(model="llama3")
llm = Ollama(model="llama3")

# ---------------- FUNCTIONS ----------------
def save_and_process(uploaded_file):
    # Create a folder for this document
    doc_name = Path(uploaded_file.name).stem
    doc_dir = os.path.join(DATA_DIR, doc_name)
    os.makedirs(doc_dir, exist_ok=True)

    # Save the uploaded file
    file_path = os.path.join(doc_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load file
    if uploaded_file.name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif uploaded_file.name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Create a separate DB for this document
    db_dir = os.path.join(doc_dir, "db")
    os.makedirs(db_dir, exist_ok=True)
    vectorstore = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
    vectorstore.persist()

    return f"✅ {uploaded_file.name} added to its own database!", vectorstore

def ask_question(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
    return qa_chain.run(query)

# ---------------- UI ----------------
st.title("📚 RAG Document Assistant (Per-Document DB)")
st.write("Upload a document, then ask questions about it. Each document has its own mini-database.")

# File upload
uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt"])
if uploaded_file is not None:
    msg, vectorstore = save_and_process(uploaded_file)
    st.success(msg)

    # Question input
    query = st.text_input("Ask a question about the uploaded doc:")
    if query:
        with st.spinner("Thinking..."):
            answer = ask_question(query, vectorstore)
            st.write("**Answer:**", answer)
