import os
import logging
from typing import List, Dict, Any

try:
    from ragatouille import RAGPretrainedModel
except ImportError:
    pass

from .cross_encoder import BaseReranker

logger = logging.getLogger(__name__)

class ColBERTReranker(BaseReranker):
    """
    Handles extremely high precision reranking via ColBERT v2.
    """
    
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0"):
        logger.info("Loading ColBERT model (%s) for ultra-precise reranking...", model_name)
        try:
            # Note: For production, you might load from a local cache or a quantized version
            self.model = RAGPretrainedModel.from_pretrained(model_name)
        except Exception as e:
            logger.error("Failed to load ColBERT model: %s", str(e))
            self.model = None

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Reranks initial retrieved hits using token-level ColBERT MaxSim interactions."""
        if not hits:
            return []
            
        if not self.model:
            logger.warning("ColBERT offline. Bypassing reranker.")
            return hits[:top_k]
            
        try:
            texts = [hit.get("chunk_text", "") for hit in hits]
            
            # Ragatouille output is typically [{ 'content': '...', 'score': 12.3, 'rank': 1, 'document_id': ... }, ...]
            scores = self.model.predict(query, texts)
            
            # Reconstruct the payload based on scores
            # Some versions of predict return just lists of floats if passing raw texts, or dicts
            # Assuming it returns a list of dictionaries with 'score' or just floats
            
            ranked_hits = []
            if scores and isinstance(scores[0], dict):
                # Mapping back to our hits
                for i, score_obj in enumerate(scores):
                    original_idx = score_obj.get("document_id")  # Depending on how predict maps them
                    # Or zip them if order is preserved or explicitly provided
                    pass
            
            # To handle robustly across ragatouille versions, if output is just floats:
            if scores and isinstance(scores[0], (int, float)):
                paired = list(zip(hits, scores))
                paired.sort(key=lambda x: x[1], reverse=True)
                
                for i, (hit, score) in enumerate(paired):
                    new_hit = hit.copy()
                    new_hit["colbert_score"] = float(score)
                    ranked_hits.append(new_hit)
            elif scores and isinstance(scores[0], dict):
                # If they mapped back, assuming content matches exactly
                content_to_hit = {h.get("chunk_text", ""): h for h in hits}
                for scr in scores:
                    content = scr.get("content", "")
                    if content in content_to_hit:
                        new_hit = content_to_hit[content].copy()
                        new_hit["colbert_score"] = float(scr.get("score", 0.0))
                        ranked_hits.append(new_hit)
            else:
                logger.warning("Unrecognized ColBERT output format. Bypassing.")
                return hits[:top_k]
                
            return ranked_hits[:top_k]
            
        except Exception as e:
            logger.error("Error during ColBERT reranking: %s", str(e))
            return hits[:top_k]
