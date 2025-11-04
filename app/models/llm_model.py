# Updated llm_model.py - Using current Cohere models

import os
from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
def load_llm():
    """Loads an LLM from ChatOpenAI using current available models"""
    # Use command-a-03-2025 (most performant) or command-r-08-2024
    llm = ChatOpenAI(model="gpt-4o-mini",            # or "gpt-4.1" for cost-efficiency
        temperature=0.3,
        max_tokens=512
    )
    return llm