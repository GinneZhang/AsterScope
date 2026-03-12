"""
Table-Specific Retrieval Path for NovaSearch.

Provides a dedicated search path for structured table data,
prioritizing chunks with table metadata and applying table-aware
reranking logic.
"""

import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class TableRetriever:
    """
    Dedicated retrieval path for table chunks.
    Filters on metadata.type == 'table' and applies table-aware
    scoring that treats Markdown structure as structured data.
    """
    
    TABLE_QUERY_INDICATORS = [
        "compare", "table", "column", "row", "values", "total",
        "average", "sum", "count", "list", "breakdown", "statistics",
        "entries", "records", "data", "metrics", "figures"
    ]
    
    def __init__(self):
        self.model = None
        self.conn = None
        
        if SentenceTransformer:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning(f"TableRetriever: Failed to load model: {e}")
        
        try:
            self.conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                port=int(os.getenv("POSTGRES_PORT", "5432")),
                dbname=os.getenv("POSTGRES_DB", "novasearch"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "")
            )
            logger.info("TableRetriever: Connected to PostgreSQL")
        except Exception as e:
            logger.warning(f"TableRetriever: No PostgreSQL connection: {e}")
    
    @staticmethod
    def is_table_query(query: str) -> bool:
        """
        Heuristic to determine if a query targets structured/tabular data.
        """
        query_lower = query.lower()
        return any(indicator in query_lower for indicator in TableRetriever.TABLE_QUERY_INDICATORS)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search specifically for table-type chunks using vector similarity.
        Only returns chunks that were marked as table content during ingestion.
        """
        if not self.model or not self.conn:
            return []
        
        try:
            query_embedding = self.model.encode(query).tolist()
            
            cur = self.conn.cursor()
            # Search only table-type chunks using PGVector cosine distance
            cur.execute("""
                SELECT id, doc_id, chunk_index, chunk_text, metadata,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE metadata->>'type' = 'table'
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))
            
            results = []
            for row in cur.fetchall():
                results.append({
                    "id": row[0],
                    "doc_id": row[1],
                    "chunk_index": row[2],
                    "chunk_text": row[3],
                    "metadata": row[4],
                    "score": float(row[5]) if row[5] else 0.0,
                    "source": "table_retriever"
                })
            
            cur.close()
            
            if results:
                logger.info(f"TableRetriever: Found {len(results)} table chunks for query.")
            return results
            
        except Exception as e:
            logger.error(f"TableRetriever search failed: {e}")
            return []
    
    def rerank_for_tables(self, query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Table-aware reranking: boosts chunks with table metadata
        and applies structural scoring for Markdown table content.
        """
        for hit in hits:
            base_score = hit.get("score", 0.0)
            text = hit.get("chunk_text", "")
            metadata = hit.get("metadata", {})
            
            # Boost factor for confirmed table chunks
            if isinstance(metadata, dict) and metadata.get("type") == "table":
                hit["score"] = base_score * 1.3  # 30% boost for table chunks
            
            # Additional boost if the chunk contains Markdown table syntax
            if "| " in text and "---" in text:
                hit["score"] = hit.get("score", base_score) * 1.1  # 10% extra for Markdown tables
        
        # Re-sort by updated score
        hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return hits
