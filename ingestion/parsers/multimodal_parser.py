import os
import logging
from typing import List, Dict, Any, Optional

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pass

try:
    from openai import OpenAI
except ImportError:
    pass

logger = logging.getLogger(__name__)

class MultimodalParser:
    """
    Handles ingestion of non-text data (Images, Tables) embedded in PDFs or Docs.
    Uses OCR (Tesseract) as a baseline, and OpenAI Vision API for complex tables/figures.
    """
    
    def __init__(self, use_vision_api: bool = False):
        self.use_vision_api = use_vision_api
        if self.use_vision_api:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                self.vision_client = OpenAI(api_key=openai_api_key)
            else:
                logger.warning("OPENAI_API_KEY missing. Falling back to simple OCR.")
                self.vision_client = None
        else:
            self.vision_client = None

    def _extract_text_tesseract(self, image_path: str) -> str:
        """Uses local pytesseract to perform basic OCR on an image."""
        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract OCR failed on {image_path}: {e}")
            return ""

    def _extract_text_vision_api(self, image_path: str) -> str:
        """Uses OpenAI Vision API to extract highly structured text/tables from an image."""
        if not self.vision_client:
            return self._extract_text_tesseract(image_path)
            
        try:
            import base64
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
            response = self.vision_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all text, data, and tables from this image perfectly as markdown. Describe any important visual diagrams briefly."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
                        ]
                    }
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content.strip() # type: ignore
        except Exception as e:
            logger.error(f"Vision API extraction failed on {image_path}: {e}")
            # Fallback
            return self._extract_text_tesseract(image_path)

    def parse_document(self, file_path: str) -> str:
        """
        Parses a document. In a real system, this would iterate through
        PDF pages, extracting text and routing images to the multimodal extractors.
        For now, this handles direct image paths.
        """
        logger.info(f"Parsing multimodal document: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in {'.png', '.jpg', '.jpeg', '.tiff'}:
            if self.use_vision_api:
                return self._extract_text_vision_api(file_path)
            else:
                return self._extract_text_tesseract(file_path)
        else:
            logger.warning(f"Unsupported explicit multimodal file type: {ext}")
            return f"Mock PDF text extraction for {file_path}"
