"""
Hybrid Search Coordinator for NovaSearch.

This module is the core of the "Tri-Engine Fusion". It orchestrates:
1. Dense Retrieval (PGVector Cosine Similarity via SentenceTransformers)
2. Sparse Retrieval (PostgreSQL Full-Text Search)
3. Graph Retrieval (Neo4j Contextual Expansion from Dense/Sparse base hits)
4. Reranking (Reciprocal Rank Fusion - RRF)
"""

import os
import logging
from typing import List, Dict, Any, Tuple
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("Please install sentence-transformers: pip install sentence-transformers")

try:
    import psycopg2
    from psycopg2.extras import DictCursor
except ImportError:
    raise ImportError("Please install psycopg2-binary: pip install psycopg2-binary")

try:
    from neo4j import GraphDatabase
except ImportError:
    raise ImportError("Please install neo4j python driver: pip install neo4j")

logger = logging.getLogger(__name__)

class HybridSearchCoordinator:
    """
    Orchestrates dense, sparse, and graph search strategies, fusing the results
    to provide highly relevant, context-grounded chunks.
    """

    def __init__(
        self,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        pg_dsn: str | None = None,
        neo4j_uri: str | None = None,
        neo4j_user: str | None = None,
        neo4j_password: str | None = None
    ):
        """
        Initializes the HybridSearchCoordinator with connections and embedding models.
        """
        # 1. Load Embedding Model
        logger.info("Loading embedding model for Dense Search...")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # 2. Setup PostgreSQL (PGVector & FTS)
        self.pg_dsn = pg_dsn or os.getenv("DATABASE_URL", 
            f"dbname={os.getenv('POSTGRES_DB', 'novasearch')} "
            f"user={os.getenv('POSTGRES_USER', 'postgres')} "
            f"password={os.getenv('POSTGRES_PASSWORD', 'postgres_secure_password')} "
            f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
            f"port={os.getenv('POSTGRES_PORT', '5432')}"
        )
        try:
            self.pg_conn = psycopg2.connect(self.pg_dsn)
            logger.info("Connected to PostgreSQL for Dense/Sparse search.")
        except Exception as e:
            logger.error("Failed to connect to PostgreSQL: %s", str(e))
            self.pg_conn = None

        # 3. Setup Neo4j (Graph Retrieval)
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "neo4j_secure_password")
        
        try:
            self.neo4j_driver = GraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
            self.neo4j_driver.verify_connectivity()
            logger.info("Connected to Neo4j Knowledge Graph.")
        except Exception as e:
            logger.error("Failed to connect to Neo4j: %s", str(e))
            self.neo4j_driver = None

    def __del__(self):
        """Cleanup connections on destruction."""
        if self.pg_conn and not self.pg_conn.closed:
            self.pg_conn.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def _dense_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search on pgvector.
        Returns a list of dicts: {'chunk_id': str, 'score': float, 'text': str, ...}
        """
        if not self.pg_conn:
            logger.warning("Postgres offline. Skipping dense search.")
            return []
            
        # 1. Embed query
        query_vector = self.embedding_model.encode(query).tolist()
        
        # 2. Execute pgvector search (assuming <-> operator for L2 or <=> for Cosine)
        # Placeholder SQL - requires a 'chunks' table with a 'embedding' vector column
        sql = """
            SELECT 
                doc_id, index, chunk_text, 
                1 - (embedding <=> %s::vector) AS similarity_score
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        
        results = []
        try:
            with self.pg_conn.cursor(cursor_factory=DictCursor) as cur:
                # We need to construct the vector string representation
                vector_str = "[" + ",".join([str(x) for x in query_vector]) + "]"
                cur.execute(sql, (vector_str, vector_str, top_k))
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "doc_id": row["doc_id"],
                        "chunk_index": row["index"],
                        "chunk_text": row["chunk_text"],
                        "score": float(row["similarity_score"]),
                        "source": "dense"
                    })
        except Exception as e:
            logger.error("Error during Dense Search: %s", str(e))
            self.pg_conn.rollback() # Rollback on error
            
        return results

    def _sparse_search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Performs PostgreSQL Full-Text Search (Keyword Matching).
        """
        if not self.pg_conn:
             logger.warning("Postgres offline. Skipping sparse search.")
             return []
             
        # Placeholder SQL for Postgres FTS
        # Assuming a 'fts_tokens' tsvector column or just plainto_tsquery on text
        sql = """
            SELECT 
                doc_id, index, chunk_text, 
                ts_rank(to_tsvector('english', chunk_text), plainto_tsquery('english', %s)) AS rank_score
            FROM chunks
            WHERE to_tsvector('english', chunk_text) @@ plainto_tsquery('english', %s)
            ORDER BY rank_score DESC
            LIMIT %s;
        """
        
        results = []
        try:
            with self.pg_conn.cursor(cursor_factory=DictCursor) as cur:
                cur.execute(sql, (query, query, top_k))
                rows = cur.fetchall()
                for row in rows:
                    results.append({
                        "doc_id": row["doc_id"],
                        "chunk_index": row["index"],
                        "chunk_text": row["chunk_text"],
                        "score": float(row["rank_score"]),
                        "source": "sparse"
                    })
        except Exception as e:
            logger.error("Error during Sparse Search: %s", str(e))
            self.pg_conn.rollback()
            
        return results

    def _graph_expansion(self, base_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes the top hits from Dense/Sparse, queries Neo4j for their parent
        document and adjacent chunks (1-hop), and injects this graph context.
        """
        if not self.neo4j_driver or not base_hits:
            return base_hits
            
        enriched_hits = []
        
        # We process each hit to find its graph neighborhood
        with self.neo4j_driver.session() as session:
            for hit in base_hits:
                doc_id = hit["doc_id"]
                idx = hit["chunk_index"]
                
                # Cypher query: Find the chunk, its parent document, and 
                # immediately adjacent chunks (idx-1 and idx+1) for context windowing.
                cypher = """
                MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk {index: $idx})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(prev:Chunk {index: $idx - 1})
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(next:Chunk {index: $idx + 1})
                RETURN 
                    d.title AS doc_title,
                    d.section AS doc_section,
                    c.chunk_text AS exact_hit_text,
                    prev.chunk_text AS prev_context,
                    next.chunk_text AS next_context
                """
                
                try:
                    result = session.run(cypher, doc_id=doc_id, idx=idx).single()
                    if result:
                        graph_context = {
                            "doc_title": result["doc_title"],
                            "doc_section": result["doc_section"],
                            "prev_context": result["prev_context"],
                            "next_context": result["next_context"]
                        }
                        # Merge the graph context into the hit
                        enriched_hit = {**hit, "graph_context": graph_context}
                        enriched_hits.append(enriched_hit)
                    else:
                        # Fallback if graph node not found
                        enriched_hits.append({**hit, "graph_context": None})
                except Exception as e:
                    logger.error("Graph Expansion error for Doc %s, Idx %s: %s", doc_id, idx, str(e))
                    enriched_hits.append({**hit, "graph_context": None})
                    
        return enriched_hits

    def _reciprocal_rank_fusion(
        self, 
        dense_results: List[Dict[str, Any]], 
        sparse_results: List[Dict[str, Any]], 
        k: int = 60
    ) -> List[Dict[str, Any]]:
        """
        Fuses Dense and Sparse results using Reciprocal Rank Fusion (RRF).
        formula: RRF_score = 1 / (k + rank)
        """
        rrf_scores: Dict[Tuple[str, int], Dict[str, Any]] = {}
        
        # Process Dense
        for rank, hit in enumerate(dense_results):
            key = (hit["doc_id"], hit["chunk_index"])
            if key not in rrf_scores:
                rrf_scores[key] = {**hit, "rrf_score": 0.0, "sources": []}
            rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
            rrf_scores[key]["sources"].append("dense")
            
        # Process Sparse
        for rank, hit in enumerate(sparse_results):
             key = (hit["doc_id"], hit["chunk_index"])
             if key not in rrf_scores:
                 rrf_scores[key] = {**hit, "rrf_score": 0.0, "sources": []}
             rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
             if "sparse" not in rrf_scores[key]["sources"]:
                 rrf_scores[key]["sources"].append("sparse")
                 
        # Sort by final RRF score descending
        fused_results = list(rrf_scores.values())
        fused_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        return fused_results

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main orchestration method for Tri-Engine Fusion.
        1. Excutes Dense & Sparse retrieval in parallel (simulation here).
        2. Fuses with Reciprocal Rank Fusion.
        3. Expands top fused results using Knowledge Graph traversal.
        
        Args:
            query (str): The natural language query.
            top_k (int): Number of final contextualized chunks to return.
            
        Returns:
            List of fused, grounded chunks ready for the Context Window & LLM.
        """
        logger.info("Initiating Hybrid Search for query: '%s'", query)
        
        # 1. Base Retrieval (Fetch slightly more to fuse well)
        fetch_k = top_k * 3 
        dense_hits = self._dense_search(query, top_k=fetch_k)
        sparse_hits = self._sparse_search(query, top_k=fetch_k)
        
        # 2. Rerank / Fuse (RRF)
        fused_hits = self._reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        # Trim to final desired K before heavy graph operations
        fused_list = list(fused_hits)
        top_fused = fused_list[:top_k]  # type: ignore
        
        # 3. Knowledge Graph Expansion
        final_grounded_results = self._graph_expansion(top_fused)
        
        logger.info("Hybrid Search Complete. Yielding %s results.", len(final_grounded_results))
        return final_grounded_results
