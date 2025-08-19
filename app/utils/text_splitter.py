from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(documents):
    """Splits documents into smaller chunks for embedding"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    return text_splitter.split_documents(documents)
