import os
from langchain_community.llms import Cohere

def load_llm():
    """Loads an LLM from Hugging Face Hub"""
    llm = Cohere()
    # llm = HuggingFaceEndpoint(
    #     repo_id="google/flan-t5-base",
    #     task="text2text-generation",
    #     max_new_tokens=512,
    #     temperature=0.3,
    #     huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    # )
    return llm
