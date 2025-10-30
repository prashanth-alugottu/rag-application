# Updated llm_model.py - Using current Cohere models

import os
from langchain_cohere import ChatCohere

def load_llm():
    """Loads an LLM from Cohere using current available models"""
    # Use command-a-03-2025 (most performant) or command-r-08-2024
    llm = ChatCohere(
        model="command-a-03-2025",  # Most performant current model
        # Alternative options:
        # model="command-r-08-2024",  # Current Command R version
        # model="command-r-plus-08-2024",  # Current Command R+ version
        temperature=0.3,
        max_tokens=512
    )
    return llm