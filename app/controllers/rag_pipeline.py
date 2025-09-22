# Updated rag_pipeline.py - Using new create_retrieval_chain approach

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from app.models.llm_model import load_llm

def build_rag_chain(vectorstore):
    """Creates a modern RAG chain using create_retrieval_chain"""
    llm = load_llm()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Create a prompt template for the RAG system
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the document chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create the retrieval chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain