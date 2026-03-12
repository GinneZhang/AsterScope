"""
Structured Query Graph Parser for NovaSearch.
Converts natural language queries into semantic triplets for guided retrieval.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional
import openai

logger = logging.getLogger(__name__)

class QueryGraphParser:
    """
    Extracts semantic entities and relationships (triplets) from user queries
    to build a structured query graph.
    """

    def __init__(self, model: str = "gpt-3.5-turbo"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI(api_key=self.api_key) if self.api_key else None
        self.model = model

    def parse(self, query: str) -> List[Dict[str, str]]:
        """
        Parses a query into a list of triplets: [{"subject": "...", "relation": "...", "object": "..."}]
        """
        if not self.client:
            logger.warning("No OpenAI client for QueryGraphParser. Skipping.")
            return []

        prompt = f"""
        Extract the core semantic relationships from the following user query as a list of triplets.
        Format: Subject -> Relation -> Object
        
        Examples:
        Query: "Who is the CEO of Acme Corp?"
        Triplets: [ {{"subject": "Acme Corp", "relation": "HAS_CEO", "object": "person"}} ]
        
        Query: "Find documents about the new trade policy"
        Triplets: [ {{"subject": "documents", "relation": "ABOUT", "object": "trade policy"}} ]
        
        Output valid JSON only.
        
        User Query: {query}
        Triplets:
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            content = response.choices[0].message.content
            data = json.loads(content)
            triplets = data.get("triplets", [])
            logger.info(f"Extracted Query Graph: {triplets}")
            return triplets
        except Exception as e:
            logger.error(f"Query parsing failed: {e}")
            return []
