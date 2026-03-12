"""
Dynamic Text-to-Cypher Generator for Neo4j with Self-Healing retry loop.
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
    
    Includes a self-healing retry loop: if the generated Cypher fails
    execution, the error is fed back to the LLM for repair.
    """

    def __init__(self, model: str = "gpt-4-turbo-preview", max_retries: int = 2):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("OPENAI_API_KEY missing for CypherGenerator.")
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
        self.max_retries = max_retries

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

    def generate_with_healing(self, user_query: str, executor_fn=None) -> Optional[str]:
        """
        Generates Cypher and validates it via execution. If execution fails,
        feeds the error back to the LLM for self-healing repair.
        
        Args:
            user_query: The natural language question.
            executor_fn: A callable that takes a Cypher string and either
                         returns results or raises an Exception on failure.
        
        Returns:
            A validated Cypher query string, or None if generation/repair fails.
        """
        cypher = self.generate(user_query)
        if not cypher or not executor_fn:
            return cypher

        last_error = None
        for attempt in range(self.max_retries + 1):  # initial + retries
            try:
                # Attempt execution to validate the Cypher
                executor_fn(cypher)
                logger.info(f"Cypher validated successfully on attempt {attempt + 1}.")
                return cypher
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Cypher execution failed (attempt {attempt + 1}/{self.max_retries + 1}): {last_error}")
                
                if attempt >= self.max_retries:
                    logger.error("Max Cypher repair retries exhausted. Giving up.")
                    return None
                
                # Self-Healing: Feed error back to LLM for repair
                cypher = self._repair_cypher(user_query, cypher, last_error)
                if not cypher:
                    logger.error("LLM could not repair the Cypher query.")
                    return None
        
        return None

    def _repair_cypher(self, original_query: str, broken_cypher: str, error_message: str) -> Optional[str]:
        """
        Feeds the broken Cypher and the Neo4j error back to the LLM
        to generate a repaired query.
        """
        if not self.api_key:
            return None

        repair_prompt = f"""
        You previously generated an invalid Cypher query. Fix it.

        {self._get_schema_context()}

        Original User Question: {original_query}
        
        Broken Cypher:
        {broken_cypher}
        
        Neo4j Error:
        {error_message}
        
        Rules:
        1. Use ONLY the nodes and relationships defined in the schema.
        2. Do NOT use any writing clauses (CREATE, MERGE, SET, DELETE, etc.).
        3. Fix the syntax or schema error based on the error message above.
        4. Output ONLY the corrected Cypher query, no preamble or explanation.
        
        Repaired Cypher:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": repair_prompt}],
                temperature=0.0
            )
            repaired = response.choices[0].message.content.strip()
            repaired = repaired.replace("```cypher", "").replace("```", "").strip()
            logger.info(f"LLM repaired Cypher: {repaired[:120]}...")
            return repaired
        except Exception as e:
            logger.error(f"Cypher repair LLM call failed: {e}")
            return None
