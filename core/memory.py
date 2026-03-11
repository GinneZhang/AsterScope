"""
Redis Session Memory Manager for NovaSearch.
"""
import os
import json
import logging
from typing import List, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import redis

logger = logging.getLogger(__name__)

class RedisMemoryManager:
    """Manages multi-turn conversation history using Redis."""
    
    def __init__(self):
        """Initializes the Redis connection."""
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_password = os.getenv("REDIS_PASSWORD", None)
        
        try:
            self.client = redis.Redis(
                host=redis_host, 
                port=redis_port, 
                password=redis_password,
                decode_responses=True # Automatically decode bytes to strings
            )
            self.client.ping()
            logger.info("Connected to Redis for Semantic Memory.")
        except Exception as e:
            logger.error("Failed to connect to Redis: %s", str(e))
            self.client = None
            
        # Default TTL for session keys: 24 hours (in seconds)
        self.ttl = 60 * 60 * 24

    def add_message(self, session_id: str, role: str, content: str):
        """
        Appends a message to the session's conversation history.
        """
        if not self.client:
            return
            
        key = f"session:{session_id}:history"
        message = json.dumps({"role": role, "content": content})
        
        try:
            # RPUSH appends to the end of the list
            self.client.rpush(key, message)
            # Reset the TTL so active sessions don't expire mid-conversation
            self.client.expire(key, self.ttl)
        except Exception as e:
            logger.error("Failed to append message to Redis for session %s: %s", session_id, str(e))

    def get_history(self, session_id: str, max_turns: int = 5) -> List[Dict[str, str]]:
        """
        Retrieves the last N turns of the conversation history.
        """
        if not self.client:
            return []
            
        key = f"session:{session_id}:history"
        fetch_count = max_turns * 2
        
        try:
            # We want the LAST fetch_count items
            raw_history = self.client.lrange(key, -fetch_count, -1)
            history = [json.loads(msg) for msg in raw_history]
            return history
        except Exception as e:
            logger.error("Failed to retrieve history from Redis for session %s: %s", session_id, str(e))
            return []
