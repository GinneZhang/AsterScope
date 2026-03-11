import os
import logging
from typing import List, Dict, Any

try:
    from elasticsearch import Elasticsearch
except ImportError:
    pass

from .keyword_search import BaseSparseRetriever

logger = logging.getLogger(__name__)

class ElasticSparseRetriever(BaseSparseRetriever):
    """
    Handles robust keyword search via Elasticsearch.
    """
    
    def __init__(self, index_name: str = "novasearch_chunks"):
        self.index_name = index_name
        es_host = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")
        es_user = os.getenv("ELASTICSEARCH_USER", "elastic")
        es_password = os.getenv("ELASTICSEARCH_PASSWORD", "changeme")
        
        try:
            self.es = Elasticsearch(
                es_host,
                basic_auth=(es_user, es_password)
            )
            if self.es.ping():
                logger.info(f"Successfully connected to Elasticsearch at {es_host}")
            else:
                logger.error("Elasticsearch ping failed.")
                self.es = None
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.es = None

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Performs a BM25 keyword search on Elasticsearch."""
        if not self.es:
            logger.warning("Elasticsearch client uninitialized. Skipping sparse search.")
            return []
            
        try:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["chunk_text", "title", "section"]
                    }
                },
                "size": top_k
            }
            
            response = self.es.search(index=self.index_name, body=body)
            hits = response.get("hits", {}).get("hits", [])
            
            results = []
            for hit in hits:
                source = hit.get("_source", {})
                results.append({
                    "doc_id": source.get("doc_id", "Unknown"),
                    "chunk_index": source.get("sequence_index", 0),
                    "chunk_text": source.get("chunk_text", ""),
                    "score": hit.get("_score", 0.0),
                    "source": "sparse_elastic"
                })
                
            return results
        except Exception as e:
            logger.error("Error during Elasticsearch search: %s", str(e))
            return []
