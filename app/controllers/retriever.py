from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.models.embedding_model import load_embedding_model
from app.utils.text_splitter import split_text
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.embeddings.openai import OpenAIEmbeddings

def build_vectorstore(file_path):
    """Loads documents, splits them, creates embeddings, and stores in FAISS"""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path)

    documents = loader.load()
    chunks = split_text(documents)

    # embeddings = load_embedding_model()

    embeddings = CohereEmbeddings(user_agent="langchain")
    # embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore
