"""
Enterprise Copilot Agent Logic.

This module acts as the "brain" of NovaSearch. It orchestrates the retrieval
of context via the HybridSearchCoordinator and constructs source-grounded LLM
prompts for the OpenAI API to enforce strict adherence to enterprise facts.
"""

import os
import logging
from typing import List, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    raise ImportError("Please install openai: pip install openai")

# Assuming hybrid_search is in the project's Python path
from retrieval.hybrid_search import HybridSearchCoordinator

logger = logging.getLogger(__name__)

class EnterpriseCopilotAgent:
    """
    The main reasoning loop that connects the user's query, the Tri-Engine
    Retrieval system, and the LLM for grounded response generation.
    """

    def __init__(self, model_name: str = "gpt-4-turbo-preview"):
        """
        Initializes the Copilot agent, setting up the OpenAI client and
        the HybridSearchCoordinator.
        """
        self.model_name = model_name
        self.api_key = os.getenv("OPENAI_API_KEY", "")
        
        if not self.api_key or "placeholder" in self.api_key:
            logger.warning("Valid OPENAI_API_KEY not found. Agent will fail on generation.")
            
        self.client = OpenAI(api_key=self.api_key)
        
        # Initialize the retrieval coordinator
        logger.info("Initializing HybridSearchCoordinator in Copilot Agent...")
        self.search_coordinator = HybridSearchCoordinator()

    def _format_context(self, hits: List[Dict[str, Any]]) -> str:
        """
        Assembles retrieved chunks and graph metadata into a highly structured
        string format for the LLM prompt.
        """
        if not hits:
            return "No relevant context found."

        context_blocks = []
        for i, hit in enumerate(hits):
            # Extract basic info
            source = hit.get("source", "unknown")
            score = hit.get("score", 0.0)
            text = hit.get("chunk_text", "").strip()
            
            # Extract Graph Context
            graph_info = hit.get("graph_context")
            if graph_info:
                title = graph_info.get("doc_title", "Unknown Title")
                section = graph_info.get("doc_section", "Unknown Section")
            else:
                title = "Unknown Document"
                section = "General"

            # Format the block with strict tracking markers
            block = f"--- [Document {i+1}] ---\n"
            block += f"[Source Marker]: [Doc: {title}, Section: {section}]\n"
            block += f"[Retrieval Type]: {source.upper()} (Score: {score:.3f})\n"
            block += f"[Content]: {text}\n"
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def _build_system_prompt(self) -> str:
        """
        Constructs the rigorous anti-hallucination system prompt.
        """
        return """You are NovaSearch Copilot, a highly precise, compliance-focused enterprise AI assistant.
Your primary directive is to answer the user's query ONLY using the provided `<CONTEXT>` blocks.

STRICT GENERATION RULES (Must be followed exactly):
1. FACTUAL GROUNDING: You must base every single factual claim in your response entirely on the `<CONTEXT>`.
2. NO HALLUCINATION: If the answer cannot be confidently derived from the `<CONTEXT>`, you must explicitly state: "I don't have enough information in the provided context to answer that." Do not attempt to guess or use external pre-training knowledge.
3. INLINE CITATIONS: Every claim or fact you state MUST include an inline citation matching the `[Source Marker]` provided with the chunk. Example format: "The onboarding process requires 3 signatures [Doc: HR Manual, Section: 1.2]."
4. SYNTHESIS: If multiple `<CONTEXT>` blocks inform your answer, synthesize them logically, but ensure all distinct sources are cited.
5. TONE: Professional, concise, and definitive.

If the user greets you or asks about your capabilities, you may respond naturally, but still reinforce your reliance on grounded data.
"""

    def generate_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Executes the full reasoning loop: Retrieval -> Context Assembly -> LLM Generation.
        
        Args:
            query (str): The user's natural language question.
            top_k (int): Number of context chunks to retrieve.
            
        Returns:
            Dict containing the 'answer' string and 'source_context' used.
        """
        logger.info("Agent received query: '%s'", query)
        
        # 1. Retrieval Phase (Tri-Engine Fusion)
        logger.info("Executing retrieval phase...")
        retrieved_hits = self.search_coordinator.search(query, top_k=top_k)
        
        # 2. Context Assembly Phase
        formatted_context = self._format_context(retrieved_hits)
        
        # 3. Prompt Construction
        system_prompt = self._build_system_prompt()
        
        user_message_parts = [
            f"User Query: {query}\n\n<CONTEXT>\n",
            formatted_context,
            "\n</CONTEXT>\n\nPlease provide your grounded answer based ONLY on the context above."
        ]
        user_message = "".join(user_message_parts)
        
        # 4. LLM Generation
        logger.info("Calling OpenAI LLM (%s)...", self.model_name)
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.1, # Low temperature for factual consistency
                max_tokens=1000
            )
            answer = response.choices[0].message.content
        except Exception as e:
            logger.error("LLM Generation failed: %s", str(e))
            answer = f"Error during response generation: {str(e)}"

        logger.info("Agent reasoning loop complete.")
        
        # Return both the generated answer and the raw retrieved payload for tracing/audit
        return {
            "answer": answer,
            "source_chunks": retrieved_hits,
            "formatted_context_used": formatted_context,
        }
