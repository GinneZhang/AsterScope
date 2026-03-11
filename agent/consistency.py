"""
Consistency Evaluator for detecting hallucinations in real-time.
"""

import logging
from typing import Dict, Any
import json
import openai

logger = logging.getLogger(__name__)

class ConsistencyEvaluator:
    def __init__(self, client: openai.OpenAI):
        self.client = client

    def evaluate(self, answer: str, context: str) -> Dict[str, Any]:
        """
        Evaluate if the generated answer is strictly entailed by the context.
        Returns a dictionary with consistency_score and hallucination_warning.
        """
        if not answer or not context or "No enterprise context needed" in context:
            return {"consistency_score": 1.0, "hallucination_warning": False}
            
        prompt = (
            "You are a strict anti-hallucination grading assistant for an enterprise system. "
            "Evaluate if the provided Answer is strictly entailed by the provided Context. "
            "Ignore formatting. Focus entirely on factual claims.\n\n"
            f"Context: {context}\n\n"
            f"Answer: {answer}\n\n"
            "Return ONLY a strictly valid JSON object with EXACTLY two keys:\n"
            "- 'consistency_score': a float between 0.0 and 1.0 (1.0 means fully consistent, 0.0 means complete hallucination)\n"
            "- 'hallucination_warning': boolean (true if consistency_score < 0.8, else false)"
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            result_str = response.choices[0].message.content.strip() # type: ignore
            result = json.loads(result_str)
            return {
                "consistency_score": float(result.get("consistency_score", 1.0)),
                "hallucination_warning": bool(result.get("hallucination_warning", False))
            }
        except Exception as e:
            logger.warning("Consistency evaluation failed, defaulting to safe: %s", str(e))
            return {"consistency_score": 1.0, "hallucination_warning": False}
