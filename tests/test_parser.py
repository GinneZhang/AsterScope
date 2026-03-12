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
    # Mocking open since we don't need a real file
    with patch("builtins.open", MagicMock()):
        with pytest.raises(Exception) as excinfo:
            parser.parse_pdf("dummy.pdf")
        assert "pypdf" in str(excinfo.value).lower() or "missing" in str(excinfo.value).lower() or "not found" in str(excinfo.value).lower()

def test_parser_image_degradation(mock_tesseract_missing):
    """Test that image parsing fails gracefully if Tesseract is missing."""
    parser = MultimodalParser()
    with patch("PIL.Image.open", MagicMock()):
        with pytest.raises(Exception) as excinfo:
            parser.parse_image("dummy.png")
        assert "tesseract" in str(excinfo.value).lower() or "not found" in str(excinfo.value).lower()

def test_supported_formats():
    """Verify supported formats list is present."""
    parser = MultimodalParser()
    assert ".pdf" in parser.supported_formats
    assert ".png" in parser.supported_formats
    assert ".docx" in parser.supported_formats
