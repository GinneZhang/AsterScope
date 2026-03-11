"""
Hybrid Search Coordinator for NovaSearch.

This module orchestrates:
1. Dense Retrieval (via vector_search.py)
2. Sparse Retrieval (via keyword_search.py)
3. Graph Retrieval (Neo4j Contextual Expansion)
4. Reranking (via rrf_fusion.py)
"""

import os
import logging
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
from neo4j import GraphDatabase

from retrieval.dense.vector_search import DenseRetriever
from retrieval.sparse.keyword_search import SparseRetriever
from retrieval.reranker.rrf_fusion import reciprocal_rank_fusion

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
        Initializes the HybridSearchCoordinator with connections and retrievers.
        """
        # 1. Setup PostgreSQL (PGVector & FTS)
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

        # 2. Setup Retrievers
        self.dense_retriever = DenseRetriever(self.pg_conn, embedding_model_name)
        self.sparse_retriever = SparseRetriever(self.pg_conn)

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
        if hasattr(self, 'neo4j_driver') and self.neo4j_driver:
            self.neo4j_driver.close()

    def _graph_expansion(self, base_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes the top hits from Dense/Sparse, queries Neo4j for their parent
        document and adjacent chunks (1-hop), and injects this graph context.
        """
        if not hasattr(self, 'neo4j_driver') or not self.neo4j_driver or not base_hits:
            return base_hits
            
        enriched_hits = []
        
        with self.neo4j_driver.session() as session:
            for hit in base_hits:
                doc_id = hit["doc_id"]
                idx = hit["chunk_index"]
                
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
                        enriched_hit = {**hit, "graph_context": graph_context}
                        enriched_hits.append(enriched_hit)
                    else:
                        enriched_hits.append({**hit, "graph_context": None})
                except Exception as e:
                    logger.error("Graph Expansion error for Doc %s, Idx %s: %s", doc_id, idx, str(e))
                    enriched_hits.append({**hit, "graph_context": None})
                    
        return enriched_hits

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Main orchestration method for Tri-Engine Fusion.
        """
        logger.info("Initiating Hybrid Search for query: '%s'", query)
        
        # 1. Base Retrieval
        fetch_k = top_k * 3 
        dense_hits = self.dense_retriever.search(query, top_k=fetch_k)
        sparse_hits = self.sparse_retriever.search(query, top_k=fetch_k)
        
        # 2. Rerank / Fuse (RRF)
        fused_hits = reciprocal_rank_fusion(dense_hits, sparse_hits)
        
        # Trim to final desired K before heavy graph operations
        fused_list = list(fused_hits)
        top_fused = fused_list[:top_k]  # type: ignore
        
        # 3. Knowledge Graph Expansion
        final_grounded_results = self._graph_expansion(top_fused)
        
        logger.info("Hybrid Search Complete. Yielding %s results.", len(final_grounded_results))
        return final_grounded_results
