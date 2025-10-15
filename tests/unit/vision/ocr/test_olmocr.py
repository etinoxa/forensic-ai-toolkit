# tests/unit/vision/ocr/test_olmocr.py
"""
Unit tests for OlmOCR engine.
Tests the integration with the project's design patterns.
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from PIL import Image
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[4] / "src"))

from fait.vision.ocr.models.olmocr_engine import OlmOCREngine
from fait.vision.ocr.base import OcrResult


class TestOlmOCREngine:
    """Test suite for OlmOCR engine."""

    def test_init_default(self):
        """Test initialization with default settings."""
        engine = OlmOCREngine()

        assert engine.name == "olmocr"
        assert engine.lang is None
        assert engine.model_id == "allenai/olmOCR-7B-0725"
        assert engine._model is None  # Lazy loading
        assert engine._processor is None

    def test_init_custom_model(self):
        """Test initialization with custom model ID."""
        custom_id = "custom/olmo-model"
        engine = OlmOCREngine(model_id=custom_id)

        assert engine.model_id == custom_id

    def test_init_from_env(self, monkeypatch):
        """Test model ID override from environment variable."""
        monkeypatch.setenv("FAIT_OLMOCR_MODEL", "env/olmo-model")
        engine = OlmOCREngine()

        assert engine.model_id == "env/olmo-model"

    def test_lazy_loading(self):
        """Test that model is not loaded until needed."""
        engine = OlmOCREngine()

        # Model should not be loaded yet
        assert engine._model is None
        assert engine._processor is None

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_ensure_loaded(self, mock_model_class, mock_processor_class):
        """Test lazy loading mechanism."""
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()
        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Mock the model's to() and eval() methods
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        engine = OlmOCREngine()
        engine._ensure_loaded()

        # Verify model was loaded
        assert engine._model is not None
        assert engine._processor is not None

        # Verify from_pretrained was called with correct args
        mock_processor_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_recognize_success(self, mock_model_class, mock_processor_class):
        """Test successful text recognition."""
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()

        # Mock the processor call
        mock_processor.return_value = {
            'pixel_values': Mock(),
            'input_ids': Mock()
        }
        mock_processor.batch_decode.return_value = ["Extracted text from image"]

        # Mock the model
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = [[101, 102, 103]]  # Mock token IDs

        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Create test image
        img = Image.new('RGB', (100, 100), color='white')

        # Test recognition
        engine = OlmOCREngine()
        result = engine.recognize(img)

        # Verify result
        assert result is not None
        assert isinstance(result, OcrResult)
        assert result.text == "Extracted text from image"
        assert result.engine == "olmocr"
        assert result.lang is None
        assert result.confidence is None  # OLMo doesn't provide confidence

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_recognize_empty_result(self, mock_model_class, mock_processor_class):
        """Test recognition when no text is found."""
        # Setup mocks
        mock_processor = Mock()
        mock_model = Mock()

        mock_processor.return_value = {
            'pixel_values': Mock(),
            'input_ids': Mock()
        }
        mock_processor.batch_decode.return_value = [""]  # Empty text

        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = [[]]

        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        img = Image.new('RGB', (100, 100))
        engine = OlmOCREngine()
        result = engine.recognize(img)

        # Should return None for empty text
        assert result is None

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_recognize_with_prompt_in_output(self, mock_model_class, mock_processor_class):
        """Test that prompt is removed from output if present."""
        mock_processor = Mock()
        mock_model = Mock()

        # Mock output that includes the prompt
        mock_processor.return_value = {
            'pixel_values': Mock(),
            'input_ids': Mock()
        }
        mock_processor.batch_decode.return_value = [
            "Extract all text from this image: The actual text content"
        ]

        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = [[101, 102]]

        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        img = Image.new('RGB', (100, 100))
        engine = OlmOCREngine()
        result = engine.recognize(img)

        # Prompt should be stripped
        assert result is not None
        assert result.text == "The actual text content"

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_ocr_interface(self, mock_model_class, mock_processor_class):
        """Test the OCR interface method."""
        mock_processor = Mock()
        mock_model = Mock()

        mock_processor.return_value = {'pixel_values': Mock(), 'input_ids': Mock()}
        mock_processor.batch_decode.return_value = ["Test text"]

        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = [[101]]

        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        img = Image.new('RGB', (100, 100))
        engine = OlmOCREngine()

        # Test OCR method (should delegate to recognize)
        result = engine.ocr(img, lang="en")

        assert result is not None
        assert result.text == "Test text"
        # Lang parameter should be ignored for OLMo

    def test_device_selection_cpu(self, monkeypatch):
        """Test device selection when CUDA is not available."""
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)

        engine = OlmOCREngine()
        assert engine.device == "cpu"

    def test_device_selection_cuda(self, monkeypatch):
        """Test device selection when CUDA is available."""
        import torch
        monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)

        engine = OlmOCREngine()
        assert engine.device == "cuda"

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_recognize_handles_numpy_array(self, mock_model_class, mock_processor_class):
        """Test that recognize can handle numpy arrays."""
        mock_processor = Mock()
        mock_model = Mock()

        mock_processor.return_value = {'pixel_values': Mock(), 'input_ids': Mock()}
        mock_processor.batch_decode.return_value = ["Array text"]

        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_model.generate.return_value = [[101]]

        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        # Create numpy array as input
        arr = np.zeros((100, 100, 3), dtype=np.uint8)

        engine = OlmOCREngine()
        result = engine.recognize(arr)

        assert result is not None
        assert result.text == "Array text"

    @patch('fait.vision.ocr.models.olmocr_engine.AutoProcessor')
    @patch('fait.vision.ocr.models.olmocr_engine.AutoModelForVision2Seq')
    def test_recognize_error_handling(self, mock_model_class, mock_processor_class):
        """Test error handling during recognition."""
        mock_processor = Mock()
        mock_model = Mock()

        # Make generate raise an exception
        mock_model.generate.side_effect = RuntimeError("GPU OOM")
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model

        mock_model_class.from_pretrained.return_value = mock_model
        mock_processor_class.from_pretrained.return_value = mock_processor

        img = Image.new('RGB', (100, 100))
        engine = OlmOCREngine()

        # Should return None on error, not raise
        result = engine.recognize(img)
        assert result is None


class TestOlmOCRIntegration:
    """Integration tests for OlmOCR with the engine factory."""

    def test_engine_factory_registration(self):
        """Test that OlmOCR is registered in the engine factory."""
        from fait.vision.ocr.models import ENGINE_ALIASES

        assert "olmocr" in ENGINE_ALIASES

    def test_get_engine_olmocr(self):
        """Test getting OlmOCR via the factory."""
        from fait.vision.ocr.models import get_engine

        engine = get_engine("olmocr")

        # Should return a lazy proxy
        assert hasattr(engine, 'ocr')
        assert hasattr(engine, 'recognize')
        assert engine.name == "olmocr"

    def test_get_engine_with_config(self):
        """Test getting OlmOCR with custom config."""
        from fait.vision.ocr.models import get_engine

        cfg = {"model_id": "custom/model", "enabled": True}
        engine = get_engine("olmocr", cfg)

        assert engine.name == "olmocr"


@pytest.mark.slow
@pytest.mark.integration
class TestOlmOCRRealModel:
    """
    Integration tests with real model.
    These tests require actual model download and GPU/CPU resources.
    Mark as slow to skip in regular CI.
    """

    @pytest.mark.skipif(
        not Path("datasets/images/text_ocr").exists(),
        reason="Test images not available"
    )
    def test_real_model_inference(self):
        """Test with real model on actual image."""
        pytest.skip("Skipping real model test - requires 7B model download")

        # Uncomment below to run with real model
        # from fait.vision.ocr.models.olmocr_engine import OlmOCREngine
        # from PIL import Image
        #
        # engine = OlmOCREngine()
        # img_path = Path("datasets/images/text_ocr/sample.jpg")
        # img = Image.open(img_path)
        #
        # result = engine.recognize(img)
        #
        # assert result is not None
        # assert len(result.text) > 0
        # assert result.engine == "olmocr"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])