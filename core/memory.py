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

    def add_message(self, session_id: str, role: str, content: str, user_id: str = "default_user"):
        """
        Appends a message to the session's conversation history.
        Also performs lightweight Cross-Session Thread Linking by tagging the first user query.
        """
        if not self.client:
            return
            
        key = f"session:{session_id}:history"
        
        try:
            # If this is the absolute first user message in this session, map it to the user's cross-session index
            if role == "user" and self.client.llen(key) == 0:
                # Naive topic extraction: first 4 words of the query (or we could use LLM/NER here)
                words = content.split()[:4]
                topic = " ".join(words).lower().strip()
                if topic:
                    # Map the session to the topic
                    mapping_key = f"user:{user_id}:sessions"
                    self.client.hset(mapping_key, session_id, topic)
        
            message = json.dumps({"role": role, "content": content})
            # RPUSH appends to the end of the list
            self.client.rpush(key, message)
            # Reset the TTL so active sessions don't expire mid-conversation
            self.client.expire(key, self.ttl)
        except Exception as e:
            logger.error("Failed to append message to Redis for session %s: %s", session_id, str(e))

    def get_related_sessions(self, user_id: str, topic: str) -> List[str]:
        """
        Fetches past session IDs that are related by thematic topic linkage.
        """
        if not self.client:
            return []
            
        mapping_key = f"user:{user_id}:sessions"
        try:
            all_sessions = self.client.hgetall(mapping_key)
            related = []
            normalized_topic = topic.lower().strip()
            for sid, t in all_sessions.items():
                if normalized_topic in t or t in normalized_topic:
                    related.append(sid)
            return related
        except Exception as e:
            logger.error("Failed to fetch related sessions for user %s: %s", user_id, str(e))
            return []

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
