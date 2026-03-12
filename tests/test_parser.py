"""
Unit tests for MultimodalParser gracefully degrading.
"""

import pytest
from unittest.mock import MagicMock, patch
from ingestion.parsers.multimodal_parser import MultimodalParser

@pytest.fixture
def mock_pypdf_missing():
    with patch("pypdf.PdfReader", side_effect=ImportError):
        yield

@pytest.fixture
def mock_tesseract_missing():
    with patch("pytesseract.image_to_string", side_effect=Exception("Tesseract not found")):
        yield

def test_parser_pdf_degradation(mock_pypdf_missing):
    """Test that PDF parsing fails gracefully if pypdf is missing."""
    parser = MultimodalParser()
    # Replace globals check logic or mock the internal check
    with patch("ingestion.parsers.multimodal_parser.globals", return_value={}):
        result = parser.parse(b"dummy pdf content", "application/pdf")
        assert result == ""

def test_parser_image_degradation(mock_tesseract_missing):
    """Test that image parsing fails gracefully if Tesseract is missing."""
    parser = MultimodalParser()
    # Force tesseract_available to False for the test
    parser.tesseract_available = False
    result = parser.parse(b"dummy image content", "image/png")
    assert result == ""

def test_mime_routing():
    """Verify that different MIME types are routed correctly (mocked)."""
    parser = MultimodalParser()
    with patch.object(parser, "_parse_docx", return_value="docx text"):
        result = parser.parse(b"dummy", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        assert result == "docx text"
