import os
import json
import logging
from typing import List, Dict, Any

try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    pass

from .vector_search import BaseDenseRetriever

logger = logging.getLogger(__name__)

class FAISSDenseRetriever(BaseDenseRetriever):
    """
    Handles semantic vector similarity search via local FAISS index.
    Maintains embeddings in memory and metadata in a dual-store JSON for quick retrieval.
    """
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", index_path: str = "faiss_index"):
        self.index_path = index_path
        self.metadata_path = f"{index_path}_meta.json"
        
        logger.info("Loading embedding model for FAISS Search...")
        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load sentence transformer for FAISS: {e}")
            self.embedding_model = None
            self.dimension = 384 # Default for MiniLM
            
        self.index = None
        self.metadata = {}
        self._load_index()

    def _load_index(self):
        """Loads FAISS index and metadata from disk if they exist, otherwise creates new."""
        try:
            if os.path.exists(f"{self.index_path}.bin"):
                self.index = faiss.read_index(f"{self.index_path}.bin")
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors.")
                
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'r') as f:
                        self.metadata = json.load(f)
            else:
                # Assuming L2 distance for exact search
                self.index = faiss.IndexFlatL2(self.dimension)
                logger.info("Created new empty FAISS index.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            self.index = faiss.IndexFlatL2(self.dimension)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs vector similarity search on FAISS index."""
        if not self.index or self.index.ntotal == 0 or not self.embedding_model:
            logger.warning("FAISS index is empty or offline. Skipping dense search.")
            return []
            
        try:
            # 1. Embed query (Faiss expects 2D array of float32)
            query_vector = self.embedding_model.encode([query]).astype('float32') # type: ignore
            
            # 2. Search index
            distances, indices = self.index.search(query_vector, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1: # FAISS returns -1 for empty slots
                    str_idx = str(idx) # JSON dict keys are strings
                    if str_idx in self.metadata:
                        meta = self.metadata[str_idx]
                        
                        # Faiss L2 distance: lower is better. We invert it for a mock "similarity score"
                        # to match pgvector's expectation where higher means more similar.
                        # This is a heuristic translation.
                        raw_dist = float(distances[0][i])
                        sim_score = 1.0 / (1.0 + raw_dist) 
                        
                        results.append({
                            "doc_id": meta.get("doc_id"),
                            "chunk_index": meta.get("index"),
                            "chunk_text": meta.get("chunk_text"),
                            "score": sim_score,
                            "source": "dense_faiss"
                        })
            
            # Sort just in case, though FAISS returns sorted
            results.sort(key=lambda x: x["score"], reverse=True)
            return results
            
        except Exception as e:
            logger.error("Error during FAISS Search: %s", str(e))
            return []
