# 🧠 DocuMind AI

**Conversational Document Intelligence using Retrieval-Augmented Generation (RAG)**

DocuMind AI is a Streamlit application that lets users upload PDF, TXT, and DOCX documents and ask natural-language questions about their content. The application uses a RAG pipeline to retrieve relevant document chunks before generating an answer with an OpenAI chat model.

## ✨ Features
- PDF, TXT and DOCX ingestion
- Recursive text chunking
- OpenAI embeddings
- ChromaDB vector store
- MMR retrieval for diverse relevant context
- History-aware query reformulation for follow-up questions
- Grounded prompting to reduce hallucinations
- Source/page metadata display
- Streamlit chat interface

## 🏗️ Architecture

```text
Documents
   ↓
Document Loaders
   ↓
Recursive Character Text Splitter
   ↓
OpenAI Embeddings
   ↓
ChromaDB
   ↓
MMR Retriever
   ↓
History-Aware Query Reformulation
   ↓
Retrieved Context + User Query
   ↓
OpenAI Chat Model
   ↓
Grounded Answer + Sources
```

## 🛠️ Tech Stack

Python · LangChain · OpenAI · ChromaDB · Streamlit · RAG

## 🚀 Run Locally

### 1. Clone
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DocuMind-AI
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure the API key
Copy `.env.example` to `.env` and add your OpenAI API key, or enter it directly in the Streamlit sidebar. **Never commit `.env` or API keys.**

### 5. Start the app
```bash
streamlit run documind_app.py
```

## 📌 Project Highlights

- **Chunking:** RecursiveCharacterTextSplitter with configurable chunk size and overlap.
- **Embeddings:** OpenAI `text-embedding-3-small`.
- **Retrieval:** ChromaDB with MMR (`k=4`, `fetch_k=12`) to balance relevance and diversity.
- **Conversational RAG:** Previous turns are used to reformulate follow-up questions into standalone search queries.
- **Grounding:** The generation prompt instructs the model to answer only from retrieved context and explicitly state when the answer is unavailable.
- **Traceability:** Retrieved document names and PDF page numbers are shown under Sources.

## 📂 Repository Structure

```text
DocuMind-AI/
├── documind_app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── data/
    └── docs/
        ├── RAG_architecture.png
        └── screenshots/
```

## 🎯 Resume Description

**DocuMind AI | Python, LangChain, OpenAI, ChromaDB, Streamlit** — Built a conversational RAG chatbot for document question answering using recursive chunking, OpenAI embeddings, ChromaDB vector retrieval, MMR search, history-aware query reformulation, and grounded LLM prompting with source/page traceability.

## 👩‍💻 Author

**Pooja Singh**

This repository represents the portfolio implementation and modifications made for learning and demonstrating practical Generative AI and RAG engineering concepts.
