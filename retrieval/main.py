"""
Dedicated FastAPI entry point for the AsterScope retrieval service.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.auth import get_api_key
from retrieval.hybrid_search import HybridSearchCoordinator

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AsterScope Retrieval API",
    description="Dedicated retrieval and reranking service for AsterScope",
    version="1.0.0",
)

_retriever: Optional[HybridSearchCoordinator] = None
_retriever_lock = threading.Lock()


class RetrievalQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language retrieval query.")
    top_k: int = Field(5, description="Number of grounded chunks to return.")
    additional_queries: Optional[List[str]] = Field(
        default=None,
        description="Optional additional retrieval queries for expansion or recall.",
    )
    query_graph: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="Optional query graph triplets for graph-aware retrieval.",
    )
    include_follow_ups: bool = Field(
        True,
        description="Whether to allow retrieval-generated follow-up queries.",
    )


class RetrievalHit(BaseModel):
    doc_id: Optional[str] = None
    chunk_index: Optional[int] = None
    title: Optional[str] = None
    chunk_text: str = ""
    score: float = 0.0
    source: Optional[str] = None
    source_type: Optional[str] = None
    graph_context: Optional[Dict[str, Any]] = None


class RetrievalSearchResponse(BaseModel):
    query: str
    top_k: int
    results: List[RetrievalHit]
    debug_metrics: Dict[str, Any] = Field(default_factory=dict)


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _get_retriever() -> HybridSearchCoordinator:
    global _retriever
    if _retriever is not None:
        return _retriever
    with _retriever_lock:
        if _retriever is None:
            logger.info("Initializing dedicated retrieval service coordinator...")
            _retriever = HybridSearchCoordinator()
    return _retriever


def _serialize_hit(hit: Dict[str, Any]) -> RetrievalHit:
    return RetrievalHit(
        doc_id=hit.get("doc_id"),
        chunk_index=hit.get("chunk_index"),
        title=hit.get("title") or (hit.get("graph_context") or {}).get("doc_title"),
        chunk_text=str(hit.get("chunk_text") or hit.get("content") or ""),
        score=_safe_float(
            hit.get("final_rank_score"),
            _safe_float(hit.get("cross_encoder_score"), _safe_float(hit.get("score", 0.0))),
        ),
        source=hit.get("source"),
        source_type=hit.get("source_type"),
        graph_context=hit.get("graph_context"),
    )


@app.get("/health")
def health() -> Dict[str, Any]:
    try:
        retriever = _get_retriever()
        return {
            "status": "ok",
            "service": "AsterScope Retrieval API",
            "graph_retrieval_enabled": bool(getattr(retriever, "enable_graph_retrieval", False)),
            "postgres_connected": bool(getattr(retriever, "pg_conn", None)) and not bool(getattr(retriever.pg_conn, "closed", 1)),
            "neo4j_connected": bool(getattr(retriever, "neo4j_driver", None)),
        }
    except Exception as exc:
        logger.error("Retrieval service health check failed: %s", str(exc))
        raise HTTPException(status_code=503, detail=f"Retrieval runtime unavailable: {exc}")


@app.post("/search", response_model=RetrievalSearchResponse, dependencies=[Depends(get_api_key)])
def search(request: RetrievalQueryRequest) -> RetrievalSearchResponse:
    try:
        retriever = _get_retriever()
        candidate_pool = retriever.collect_candidate_pool(
            request.query,
            top_k=request.top_k,
            additional_queries=request.additional_queries,
            include_follow_ups=request.include_follow_ups,
        )
        final_hits = retriever.finalize_candidates(
            request.query,
            candidate_pool,
            top_k=request.top_k,
            query_graph=request.query_graph,
        )
        return RetrievalSearchResponse(
            query=request.query,
            top_k=request.top_k,
            results=[_serialize_hit(hit) for hit in final_hits],
            debug_metrics=dict(retriever.last_search_debug or {}),
        )
    except Exception as exc:
        logger.error("Retrieval service /search failed: %s", str(exc))
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")
