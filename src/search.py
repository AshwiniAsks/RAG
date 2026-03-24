import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

class RAGSearch:
    def __init__(self, vectorstore: FaissVectorStore, llm_model: str = "llama-3.1-8b-instant"):
        self.vectorstore = vectorstore
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        self.llm = ChatGroq(api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] GROQ LLM initialized with model: {llm_model}")
        
    def search_and_summarize(self, query: str, top_k: int =5 )-> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant information found in the documents."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content
    