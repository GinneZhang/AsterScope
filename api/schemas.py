"""
Pydantic Schemas for NovaSearch API.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Incoming request to the NovaSearch /ask endpoint.
    """
    query: str = Field(..., description="The natural language question to ask the Copilot.")
    session_id: Optional[str] = Field(default=None, description="Optional UUID for conversation history. Generated if blank.")
    top_k: int = Field(5, description="Number of contextual chunks to retrieve before fusion and graph expansion.")


class SourceChunk(BaseModel):
    """
    Represents a grounded source piece of context used by the LLM.
    """
    doc_id: str
    chunk_index: int
    chunk_text: str
    score: float
    source: str
    graph_context: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """
    Outgoing response from the NovaSearch /ask endpoint.
    """
    answer: str = Field(..., description="The source-grounded generated answer from the LLM.")
    session_id: str = Field(..., description="UUID returned to client to continue the conversation.")
    sources: List[SourceChunk] = Field(default_factory=list, description="The raw retrieved chunks used as basis for the answer.")


class DocumentUploadRequest(BaseModel):
    """
    Placeholder schema for future ingestion endpoints.
    """
    document_text: str = Field(..., description="Raw text of the document.")
    title: str = Field(..., description="Document Title.")
    section: str = Field(default="General", description="Section or Category of the document.")
