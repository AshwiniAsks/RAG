from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

if __name__ == "__main__":
    vectorstore = FaissVectorStore("faiss_store")

    if not vectorstore.exists():
        print("Building vector store from documents...")
        docs = load_all_documents("data")
        print(f"Loaded {len(docs)} documents")
        vectorstore.build_from_documents(docs)
    else:
        print("Loading existing vector store...")
        vectorstore.load()
        
    rag_search = RAGSearch(vectorstore)
    query = "What is React and how does it work?"
    summary = rag_search.search_and_summarize(query , top_k=3)
    print("Summary:", summary)
    
        

    