"""
Unit tests for CypherGenerator.
"""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.graph.cypher_generator import CypherGenerator

@pytest.fixture
def mock_openai():
    with patch("openai.OpenAI") as mock:
        yield mock

def test_cypher_generation_success(mock_openai):
    """Test successful Cypher generation."""
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="MATCH (n:Entity) RETURN n.name"))]
    )
    
    gen = CypherGenerator()
    cypher = gen.generate("Show all entities.")
    
    assert cypher == "MATCH (n:Entity) RETURN n.name"
    assert "User Question: Show all entities." in mock_client.chat.completions.create.call_args[1]["messages"][0]["content"]

def test_cypher_not_applicable(mock_openai):
    """Test when the question is not suitable for graph traversal."""
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="NOT_APPLICABLE"))]
    )
    
    gen = CypherGenerator()
    cypher = gen.generate("Who are you?")
    
    assert cypher is None

def test_cypher_markdown_cleanup(mock_openai):
    """Test that markdown code blocks are removed from the LLM output."""
    mock_client = mock_openai.return_value
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="```cypher\nMATCH (n) RETURN n\n```"))]
    )
    
    gen = CypherGenerator()
    cypher = gen.generate("Complex query")
    
    assert cypher == "MATCH (n) RETURN n"
