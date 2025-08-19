from langchain.chains import RetrievalQA
from app.models.llm_model import load_llm

def build_rag_chain(vectorstore):
    """Creates a RetrievalQA chain"""
    llm = load_llm()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain