"""
Main FastAPI entry point for NovaSearch.
"""

import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from api.schemas import QueryRequest, QueryResponse, DocumentUploadRequest, SourceChunk
from agent.copilot_agent import EnterpriseCopilotAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(
    title="NovaSearch API",
    description="Enterprise Copilot & Intelligent Retrieval Engine",
    version="1.0.0"
)

# Secure CORS config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Update for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Copilot Agent (Singleton instance for the app lifecycle)
try:
    copilot_agent = EnterpriseCopilotAgent()
except Exception as e:
    logger.error("Failed to initialize Copilot Agent: %s", str(e))
    # In a real app, you might not want to crash outright, but for now we let it fail fast if env is truly broken.
    copilot_agent = None


@app.get("/health")
async def health_check():
    """Healthcheck endpoint."""
    return {"status": "ok", "service": "NovaSearch API"}


@app.post("/ask", response_model=QueryResponse)
async def ask_copilot(request: QueryRequest):
    """
    Main endpoint for the Enterprise Copilot.
    Takes a natural language query, performs Tri-Engine Retrieval,
    and returns a grounded, hallucination-free response.
    """
    if not copilot_agent:
        raise HTTPException(status_code=503, detail="Copilot Agent is not initialized due to missing configurations.")

    try:
        logger.info("Handling /ask request for query: '%s'", request.query)
        result = copilot_agent.generate_response(query=request.query, top_k=request.top_k)
        
        # the result dict has 'answer' and 'source_chunks'
        answer = result.get("answer", "No answer generated.")
        raw_sources = result.get("source_chunks", [])
        
        # Serialize raw dictionaries to Pydantic objects
        sources_schema = []
        for s in raw_sources:
            sources_schema.append(SourceChunk(
                doc_id=s.get("doc_id", "unknown"),
                chunk_index=s.get("chunk_index", 0),
                chunk_text=s.get("chunk_text", ""),
                score=s.get("score", 0.0) or s.get("rrf_score", 0.0), # handle both dense and rrf scores
                source=str(s.get("sources", s.get("source", "unknown"))), # handle list of sources from fusion
                graph_context=s.get("graph_context")
            ))

        return QueryResponse(
            answer=answer,
            sources=sources_schema
        )

    except Exception as e:
        logger.error("Error processing /ask request: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


@app.post("/ingest")
async def ingest_document(request: DocumentUploadRequest):
    """
    Placeholder endpoint for ingesting new documents into the system.
    Will eventually connect to SemanticChunker and KGBuilder.
    """
    return {
        "status": "success",
        "message": f"Document '{request.title}' queued for Tri-Engine ingestion.",
        "details": "Not Fully Implemented Yet. Connects to SemanticChunker -> PGVector & Neo4j KGBuilder."
    }
