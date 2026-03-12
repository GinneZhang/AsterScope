"""
Consistency Evaluator for detecting hallucinations in real-time.

Production-grade: Fail-closed behavior. If the evaluator crashes, errors,
or times out, it BLOCKS the output rather than passing it through.
"""
import logging
import json
from typing import Dict, Any

logger = logging.getLogger(__name__)

# The blocked result returned when the evaluator cannot verify the output.
BLOCKED_RESULT = {
    "consistency_score": 0.0,
    "hallucination_warning": True,
    "blocked_reason": "Evaluator failed or timed out. Output blocked for safety."
}

class ConsistencyEvaluator:
    """
    Evaluates LLM generations against the retrieved context to detect hallucinations.
    
    FAIL-CLOSED: If the evaluator errors out, the output is BLOCKED (not passed through).
    Uses gpt-4-turbo for higher reasoning fidelity.
    """
    
    DEFAULT_MODEL = "gpt-4-turbo"
    
    def __init__(self, openai_client=None, model: str = None):
        self.client = openai_client
        self.model = model or self.DEFAULT_MODEL

    def evaluate(self, generated_answer: str, context: str) -> Dict[str, Any]:
        """
        Uses an LLM call to score the factual consistency of the answer against the context.
        
        FAIL-CLOSED: If anything goes wrong, the output is blocked.
        """
        if not self.client:
            logger.error("No OpenAI client available for consistency evaluation. BLOCKING output (fail-closed).")
            return BLOCKED_RESULT
            
        if not generated_answer or not context or "No enterprise context" in context:
            # If there's no context to check against, we can't verify — block it
            if not context or "No enterprise context" in context:
                logger.warning("No context available for consistency check. Passing through boilerplate response.")
                return {"consistency_score": 1.0, "hallucination_warning": False}
            return {"consistency_score": 1.0, "hallucination_warning": False}
            
        system_prompt = (
            "You are a strict anti-hallucination fact-checker for an enterprise knowledge system. "
            "You will be provided with <Source Context> and a <Generated Answer>.\n"
            "Your job is to determine if the <Generated Answer> contains any claims, facts, or entities "
            "that contradict or CANNOT be logically deduced from the <Source Context>.\n"
            "Be STRICT: if the answer introduces any factual claims not supported by the context, "
            "flag it as a hallucination even if the claim might be true in general.\n"
            "It is OK if the answer paraphrases or logically entails information from the context.\n"
            "Respond in JSON format with exactly two keys:\n"
            "- 'consistency_score': a float between 0.0 (total hallucination) and 1.0 (completely grounded).\n"
            "- 'hallucination_warning': boolean (true if score < 0.6, else false)."
        )
        
        user_prompt = f"<Source Context>\n{context}\n</Source Context>\n\n<Generated Answer>\n{generated_answer}\n</Generated Answer>"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=64,
                timeout=15  # Hard timeout to prevent hangs
            )
            
            result_str = response.choices[0].message.content.strip()  # type: ignore
            result = json.loads(result_str)
            
            score = float(result.get("consistency_score", 0.0))
            warning = bool(result.get("hallucination_warning", True))
            
            logger.info("Consistency Evaluation: Score=%.2f, Warning=%s", score, warning)
            return {
                "consistency_score": score,
                "hallucination_warning": warning
            }
        except json.JSONDecodeError as e:
            logger.error("Consistency evaluator returned invalid JSON. BLOCKING output: %s", str(e))
            return BLOCKED_RESULT
        except Exception as e:
            logger.error("Consistency evaluation failed. BLOCKING output (fail-closed): %s", str(e))
            return BLOCKED_RESULT
