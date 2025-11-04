# Updated retriever.py - Using current Cohere embedding model

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.utils.text_splitter import split_text
from langchain_cohere import CohereEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader, PyPDFLoader, TextLoader


def build_vectorstore(file_path):
    """Loads documents, splits them, creates embeddings, and stores in FAISS"""
    if file_path.endswith(".pdf"):
        loader = safe_load_pdf(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    chunks = split_text(documents)

    # Use the current Cohere embeddings model
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def safe_load_pdf(file_path):
    # Try PyMuPDF first (best)
    try:
        return PyMuPDFLoader(file_path).load()
    except Exception:
        pass

    # Fallback to PyPDF
    try:
        return PyPDFLoader(file_path).load()
    except Exception:
        pass

    raise Exception("❌ Failed to read PDF. Try uploading a searchable PDF.")
