"""
Unit tests for FAIT Speech-to-Text service and Faster-Whisper model.
This tests low-level functionality in isolation (no CLI).
"""

import pytest
from pathlib import Path
from types import SimpleNamespace

from fait.audio.services.speech_to_text_service import SpeechToTextService
from fait.audio.speech_to_text.models.whisper_faster_model import WhisperFasterModel


# ==========================================================
# 🧩 Fake Whisper Model for mocking backend inference
# ==========================================================
class _FakeWhisperModel:
    """Fake whisper-faster backend with controlled outputs."""

    def __init__(self, *args, **kwargs):
        self.called_with = None

    def transcribe(self, audio_path, beam_size=5, vad_filter=True, task="transcribe", **extra_kwargs):
        """Simulate whisper transcribe/translate behavior."""
        self.called_with = {"audio_path": audio_path, "task": task}

        class Info:
            duration = 1.23
            language = "en" if task == "transcribe" else "ja"
            language_probability = 0.99

        class Segment:
            def __init__(self, text):
                self.text = text

        text = (
            "Hello world from fake model"
            if task == "transcribe"
            else "これは翻訳テキストです"  # “This is translated text”
        )
        segments = [Segment(text)]
        return segments, Info()


# ==========================================================
# 🧪 Fixtures
# ==========================================================
@pytest.fixture
def fake_audio(tmp_path):
    """Create a fake WAV path (no decoding required)."""
    fake_wav = tmp_path / "fake.wav"
    fake_wav.write_bytes(b"fake audio content")
    return fake_wav


@pytest.fixture
def minimal_config():
    """Provide a minimal fake config."""
    return SimpleNamespace(
        audio=SimpleNamespace(
            speech_to_text=SimpleNamespace(
                engine="whisper-faster",
                model_id="tiny",
                device="cpu",
                cache_dir=None,
            )
        )
    )


# ==========================================================
# ✅ Tests for WhisperFasterModel
# ==========================================================
def test_whisper_faster_model_transcribe(monkeypatch, fake_audio):
    """Verify WhisperFasterModel runs and returns proper metadata."""
    # Patch the LOCAL import used inside FAIT's whisper_faster_model.py
    monkeypatch.setattr(
        "fait.audio.speech_to_text.models.whisper_faster_model.WhisperModel",
        _FakeWhisperModel
    )

    model = WhisperFasterModel(model_size="tiny", device="cpu", compute_type="int8")
    text, meta = model.transcribe(fake_audio)

    assert "fake model" in text
    assert meta["language"] == "en"
    assert meta["duration"] > 0
    assert "processing_time" in meta


def test_whisper_faster_model_translate(monkeypatch, fake_audio):
    """Verify translation mode works properly (task='translate')."""
    monkeypatch.setattr(
        "fait.audio.speech_to_text.models.whisper_faster_model.WhisperModel",
        _FakeWhisperModel
    )

    model = WhisperFasterModel(model_size="tiny", device="cpu", compute_type="int8")
    text, meta = model.transcribe(fake_audio, task="translate")

    assert "翻訳" in text  # Japanese translation text
    assert meta["language"] == "ja"
    assert meta["duration"] == pytest.approx(1.23, rel=1e-3)


# ==========================================================
# ✅ Tests for SpeechToTextService
# ==========================================================
def test_speech_to_text_service_calls_pipeline(monkeypatch, fake_audio, minimal_config):
    """Ensure SpeechToTextService correctly calls the pipeline and returns results."""
    called = {}

    def fake_run_pipeline(audio_path, config):
        called["audio_path"] = audio_path
        return {
            "text": "Fake transcription complete",
            "language": "en",
            "duration": 1.23,
            "probability": 0.99,
        }

    # Patch pipeline
    monkeypatch.setattr(
        "fait.audio.pipelines.speech_to_text_pipeline.run_speech_to_text_pipeline",
        fake_run_pipeline,
    )

    service = SpeechToTextService()
    service.config = minimal_config  # set manually since __init__ doesn’t accept it
    text, meta = service.transcribe(fake_audio)

    assert "Fake transcription" in text
    assert called["audio_path"] == fake_audio
    assert meta["language"] == "en"


def test_speech_to_text_service_handles_errors(monkeypatch, fake_audio, minimal_config):
    """Confirm SpeechToTextService raises RuntimeError on failure."""
    def fake_run_pipeline(*args, **kwargs):
        raise RuntimeError("Mock pipeline failure")

    monkeypatch.setattr(
        "fait.audio.pipelines.speech_to_text_pipeline.run_speech_to_text_pipeline",
        fake_run_pipeline,
    )

    service = SpeechToTextService()
    service.config = minimal_config

    with pytest.raises(RuntimeError, match="Speech-to-Text failed"):
        service.transcribe(fake_audio)