"""
Dynamic Text-to-Cypher Generator for Neo4j.
"""

import os
import logging
from typing import Optional
import openai

logger = logging.getLogger(__name__)

class CypherGenerator:
    """
    Translates natural language questions into read-only Cypher queries
    using an LLM grounded by the Neo4j schema.
    """

    def __init__(self, model: str = "gpt-4-turbo-preview"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY missing for CypherGenerator.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model

    def _get_schema_context(self) -> str:
        """Simple schema description for the LLM."""
        return """
        Neo4j Schema:
        - Nodes:
            1. Document (id, title, section, metadata)
            2. Chunk (id, index, chunk_text)
            3. Entity (name, type)
        - Relationships:
            1. (Document)-[:HAS_CHUNK]->(Chunk)
            2. (Chunk)-[:MENTIONS]->(Entity)
        """

    def generate(self, user_query: str) -> Optional[str]:
        """
        Generates a Cypher query from a user question.
        Returns None if the question is not suitable for graph traversal.
        """
        if not self.api_key:
            return None

        prompt = f"""
        You are a Neo4j Cypher expert. Convert the following user question into a VALID, READ-ONLY Cypher query.
        
        {self._get_schema_context()}
        
        Rules:
        1. Use ONLY the nodes and relationships defined in the schema.
        2. RETURN data in a structured format (e.g., RETURN d.title, c.chunk_text, e.name).
        3. Do NOT use any writing clauses (CREATE, MERGE, SET, DELETE, etc.).
        4. If the question cannot be answered via the graph schema, respond with "NOT_APPLICABLE".
        5. Output ONLY the Cypher query or "NOT_APPLICABLE", no preamble.
        
        User Question: {user_query}
        
        Cypher:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
            )
            cypher = response.choices[0].message.content.strip()
            
            if "NOT_APPLICABLE" in cypher:
                return None
            
            # Basic cleanup (remove markdown code blocks if any)
            cypher = cypher.replace("```cypher", "").replace("```", "").strip()
            return cypher
        except Exception as e:
            logger.error(f"Cypher generation failed: {e}")
            return None
