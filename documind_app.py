import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

st.set_page_config(page_title="DocuMind AI", page_icon="🧠", layout="wide")

st.title("🧠 DocuMind AI")
st.caption("Conversational document intelligence powered by Retrieval-Augmented Generation (RAG)")

# ----------------------------- Configuration -----------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def load_uploaded_documents(uploaded_files):
    """Save uploaded files temporarily and load them as LangChain Documents."""
    documents = []
    temp_dir = Path(tempfile.mkdtemp(prefix="documind_"))

    for uploaded_file in uploaded_files:
        file_path = temp_dir / uploaded_file.name
        file_path.write_bytes(uploaded_file.getvalue())
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            docs = PyPDFLoader(str(file_path)).load()
        elif suffix == ".txt":
            docs = TextLoader(str(file_path), encoding="utf-8").load()
        elif suffix == ".docx":
            docs = Docx2txtLoader(str(file_path)).load()
        else:
            continue

        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
        documents.extend(docs)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return documents


def build_vector_store(documents, api_key):
    """Chunk documents, create embeddings, and build an in-memory Chroma store."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
    )
    return vector_store, len(chunks)


def build_rag_chain(vector_store, api_key, model_name):
    """Create a history-aware conversational RAG chain."""
    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=api_key,
    )

    # MMR retrieves relevant but less-redundant chunks.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 12},
    )

    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Given the chat history and the latest user question, rewrite the "
            "question as a standalone search query. Do not answer the question.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are DocuMind AI, a document-grounded assistant. "
            "Answer using ONLY the supplied context. If the answer is not "
            "present in the context, say that the information is not available "
            "in the uploaded documents. Do not invent facts.\n\n"
            "Context:\n{context}",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


def format_sources(documents):
    """Create a compact source list from retrieved documents."""
    seen = set()
    sources = []
    for doc in documents:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page")
        label = f"{source}"
        if page is not None:
            label += f" — page {int(page) + 1}"
        if label not in seen:
            seen.add(label)
            sources.append(label)
    return sources


# ----------------------------- Sidebar -----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Your key is used only for this application session.",
    )
    model_name = st.selectbox(
        "Chat model",
        ["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
    )

    if st.button("🚀 Build Knowledge Base", use_container_width=True):
        if not api_key:
            st.error("Please provide an OpenAI API key.")
        elif not uploaded_files:
            st.error("Please upload at least one document.")
        else:
            with st.spinner("Processing documents and creating embeddings..."):
                docs = load_uploaded_documents(uploaded_files)
                if not docs:
                    st.error("No supported document content could be loaded.")
                else:
                    vector_store, chunk_count = build_vector_store(docs, api_key)
                    st.session_state.vector_store = vector_store
                    st.session_state.rag_chain = build_rag_chain(
                        vector_store, api_key, model_name
                    )
                    st.session_state.chat_history = []
                    st.session_state.sources = []
                    st.success(f"Knowledge base ready — {chunk_count} chunks indexed.")

    if st.button("🗑️ Clear Session", use_container_width=True):
        for key in ["vector_store", "rag_chain", "chat_history", "sources"]:
            st.session_state.pop(key, None)
        st.rerun()

    st.divider()
    st.markdown("### Pipeline")
    st.markdown(
        "`Documents → Chunking → Embeddings → ChromaDB → MMR Retrieval → LLM`"
    )

# ----------------------------- Chat -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.get("rag_chain"):
    st.info("Upload your documents in the sidebar and click **Build Knowledge Base** to start chatting.")
else:
    question = st.chat_input("Ask a question about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        chat_history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant context..."):
                result = st.session_state.rag_chain.invoke(
                    {"input": question, "chat_history": chat_history}
                )
                answer = result["answer"]
                retrieved_docs = result.get("context", [])
                sources = format_sources(retrieved_docs)

            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for source in sources:
                        st.markdown(f"- {source}")

        st.session_state.messages.append({"role": "assistant", "content": answer})
