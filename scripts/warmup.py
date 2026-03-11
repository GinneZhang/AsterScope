import os
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def warmup_models():
    """
    Pre-downloads massive ML weights (SentenceTransformers, CrossEncoders, spaCy NER) 
    into the local HuggingFace/spaCy cache to prevent API timeouts on the first request.
    """
    logger.info("Initializing Machine Learning Warmup Sequence...")
    
    # 1. Base Embedding Model
    logger.info("Downloading Base Embedding Model (all-MiniLM-L6-v2)...")
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("[SUCCESS] Base Embedding Model Cached.")
    except Exception as e:
        logger.error("[FAIL] Base Embedding Model: %s", e)

    # 2. Cross-Encoder Model
    logger.info("Downloading Cross-Encoder Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    try:
        from sentence_transformers import CrossEncoder
        CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("[SUCCESS] Cross-Encoder Model Cached.")
    except Exception as e:
        logger.error("[FAIL] Cross-Encoder Model: %s", e)

    # 3. spaCy NER Model
    logger.info("Downloading spaCy NER Engine (en_core_web_sm)...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_sm")
        except OSError:
            logger.info("Model not found locally. Pulling via python -m spacy download...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
            spacy.load("en_core_web_sm")
        logger.info("[SUCCESS] spaCy NER Model Cached.")
    except Exception as e:
        logger.error("[FAIL] spaCy NER Model: %s", e)

    logger.info("Warmup Sequence Complete! The NovaSearch API is ready for instant requests.")

if __name__ == "__main__":
    warmup_models()
