"""
Unit tests for Redis Semantic Memory.
"""

import pytest
from unittest.mock import MagicMock, patch
import json
import numpy as np
from core.memory import RedisMemoryManager

@pytest.fixture
def mock_redis():
    with patch("redis.Redis") as mock:
        yield mock

@pytest.fixture
def mock_transformer():
    with patch("sentence_transformers.SentenceTransformer") as mock:
        yield mock

def test_redis_memory_init(mock_redis, mock_transformer):
    """Test initialization of RedisMemoryManager."""
    manager = RedisMemoryManager()
    assert manager.client is not None
    assert manager.model is not None

def test_add_message_with_semantic_vector(mock_redis, mock_transformer):
    """Test adding a message handles semantic vector storage on first turn."""
    mock_model = mock_transformer.return_value
    mock_model.encode.return_value = np.array([0.1, 0.2, 0.3])
    
    mock_client = mock_redis.return_value
    mock_client.llen.return_value = 0 # First message
    
    manager = RedisMemoryManager()
    manager.add_message("session_1", "user", "Hello world", user_id="user_123")
    
    # Check if vector was stored
    mock_model.encode.assert_called_with("Hello world")
    assert mock_client.hset.call_count >= 2 # Once for snippet, once for vector

def test_get_related_sessions_similarity(mock_redis, mock_transformer):
    """Test semantic similarity search for related sessions."""
    mock_model = mock_transformer.return_value
    # Query vector
    mock_model.encode.return_value = np.array([1.0, 0.0, 0.0])
    
    mock_client = mock_redis.return_value
    # Mock existing session vectors in Redis
    # Session A: identical [1,0,0], Session B: orthogonal [0,1,0]
    mock_client.hgetall.return_value = {
        "session_a": json.dumps([1.0, 0.0, 0.0]),
        "session_b": json.dumps([0.0, 1.0, 0.0])
    }
    
    manager = RedisMemoryManager()
    related = manager.get_related_sessions("user_123", "Search query", threshold=0.5)
    
    assert "session_a" in related
    assert "session_b" not in related
