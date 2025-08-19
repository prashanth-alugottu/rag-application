import streamlit as st
from app.controllers.retriever import build_vectorstore
from app.controllers.rag_pipeline import build_rag_chain
import os
from dotenv import load_dotenv


def main():
    load_dotenv()  # Load environment variables from .env file
    st.set_page_config(page_title="RAG Chatbot", layout="wide")
    st.title("🤖 RAG-based Chatbot with LangChain + Hugging Face")

    uploaded_file = st.file_uploader("Upload your document", type=["pdf", "txt"])
    if uploaded_file:
         # ensure data/ folder exists
        os.makedirs("app/data", exist_ok=True)
        file_path = f"app/data/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        os.environ["COHERE_API_KEY"] =os.getenv("COHERE_API_KEY")
        # os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        st.success("✅ Document uploaded successfully!")
        vectorstore = build_vectorstore(file_path)
        st.success("✅ Stored Vectors in Vector DB.")
        rag_chain = build_rag_chain(vectorstore)
        
        st.subheader("Ask a question:")
        query = st.text_input("Enter your query")
        if query:
            result = rag_chain({"query": query})
            st.write("### Answer:", result["result"])
            with st.expander("🔍 Source Documents"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content[:200] + "...")
            st.success("✅ Answer retrieved successfully!")
    else:
        st.info("Please upload a document to start.")
