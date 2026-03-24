import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir="faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(embedding_model)
        print(f"[INFO] Loaded embedding model: {embedding_model}")
      
        
    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} documents")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        print(f"[INFO] Created {len(chunks)} chunks from documents")
        embeddings = emb_pipe.embed_chunks(chunks)
        print(f"[INFO] Generated embeddings with shape: {embeddings.shape}")
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")
       
        
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
            print(f"[INFO] Created new FAISS index with dimension: {dim}")
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} Vectors to FAISS index. Total embeddings: {self.index.ntotal}")
    
        
    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")
      
        
    def exists(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        return os.path.exists(faiss_path) and os.path.exists(meta_path)
 
        
    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"Vector store files not found in {self.persist_dir}")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")
 
        
    def clear(self):
        """Clear the vector store"""
        self.index = None
        self.metadata = []
        print("[INFO] Vector store cleared")
 
        
    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        # Ensure query_embedding is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < len(self.metadata) and idx >= 0:
                meta = self.metadata[idx]
            else:
                meta = None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results


    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store with: {query_text}")
        query_emb = self.model.encode([query_text], show_progress_bar=False).astype('float32')
        return self.search(query_emb, top_k)