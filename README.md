# rag_doc_assistant
RAG Document Assistant is an AI-powered application that combines Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs) to provide accurate, context-aware answers from documents. Instead of relying only on the model’s memory, it retrieves relevant document chunks and uses them to generate precise responses.
---

## 🚀 Features
- 📂 Upload and process your own documents (PDF, TXT, DOCX, etc.)
- 🔍 Perform **semantic search** over documents
- 🤖 Answer questions using **RAG-powered LLMs**
- ⚡ Fast retrieval with **vector databases (FAISS/Chroma)**
- 🖥️ Simple and interactive UI (Streamlit/Gradio)

---

## 🛠️ Tech Stack
- **Python**
- **LangChain**
- **FAISS / Chroma**
- **OpenAI API / Hugging Face models**
- **Streamlit** (for UI)

---
## 📦 Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/manshul144/rag_doc_assistant.git
   cd rag_doc_assistant

2. Install Dependencies:
   pip install -r requirements.txt

3. Add your API keys (e.g., OpenAI) in a .env file:
   OPENAI_API_KEY=your_api_key_here

4. Run the app with:
   streamlit run app.py




## 📂 Project Structure
rag_doc_assistant/
│-- app.py              # Main app (Streamlit/Gradio)
│-- requirements.txt    # Dependencies
│-- README.md           # Project description
│-- data/               # Sample documents
│-- utils/              # Helper functions

