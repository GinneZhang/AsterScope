"""
MonoT5 Sequence-to-Sequence Reranker.
"""

import logging
from typing import List, Dict, Any

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class MonoT5Reranker:
    """
    Reranks documents using a sequence-to-sequence model (MonoT5).
    MonoT5 estimates the probability of the model generating the word "true"
    when prompted with "Query: [q] Document: [d] Relevant:".
    """
    
    def __init__(self, model_name: str = "castorini/monot5-base-msmarco"):
        logger.info(f"Initializing MonoT5Reranker with model {model_name}...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device).eval()
            
            # The token id for "true" or " false". Space prefixes vary by tokenizer configs.
            self.true_token_id = self.tokenizer.encode("true")[0]
            self.false_token_id = self.tokenizer.encode("false")[0]
            logger.info(f"MonoT5 loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load MonoT5 model: {e}")
            self.tokenizer = None
            self.model = None

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Reranks a list of retrieved chunks using MonoT5.
        """
        if not hits:
            return []
            
        if not self.model or not self.tokenizer:
            logger.warning("MonoT5 model not loaded. Returning original hits.")
            return hits[:top_k]

        reranked_hits = []
        batch_queries = []
        
        for hit in hits:
            doc_text = hit.get("chunk_text", "")
            # MonoT5 prompt format
            input_text = f"Query: {query} Document: {doc_text} Relevant:"
            batch_queries.append(input_text)
            
        try:
            with torch.no_grad():
                inputs = self.tokenizer(batch_queries, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
                
                # Decoder input ids (just the pad token is enough to start generation)
                decoder_input_ids = torch.full(
                    (len(batch_queries), 1),
                    self.model.config.decoder_start_token_id,
                    dtype=torch.long,
                    device=self.device
                )
                
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids
                )
                
                # Get the logits for the very first generated token across the vocabulary
                logits = outputs.logits[:, 0, :]
                
                # Get logits for 'true' and 'false' tokens specifically
                true_logits = logits[:, self.true_token_id]
                false_logits = logits[:, self.false_token_id]
                
                # Softmax over True/False
                scores = torch.nn.functional.log_softmax(torch.stack([true_logits, false_logits], dim=1), dim=1)[:, 0] # log probability of True

            for idx, hit in enumerate(hits):
                hit_copy = hit.copy()
                hit_copy["monot5_score"] = float(scores[idx].cpu().numpy())
                reranked_hits.append(hit_copy)
                
            reranked_hits.sort(key=lambda x: x["monot5_score"], reverse=True)
            return reranked_hits[:top_k]
            
        except Exception as e:
            logger.error(f"MonoT5 Reranking failed: {e}")
            return hits[:top_k]
