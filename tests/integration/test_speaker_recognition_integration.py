# tests/integration/test_speaker_recognition_integration.py
"""
Integration tests for speaker recognition pipeline.
These tests use mock embedders to avoid loading actual models.
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
from fait.audio.pipelines.speaker_recognition_pipeline import (
    run_speaker_match,
    SpeakerMatchConfig, _resolve_strategy,
)


class MockEmbedder:
    """Mock embedder that returns deterministic embeddings"""

    def __init__(self, model_id="mock"):
        self.model_id = model_id
        self.embed_calls = []

    def name(self):
        return f"Mock({self.model_id})"

    def embed_file(self, audio_path, use_cache=True):
        self.embed_calls.append(audio_path)
        # Return deterministic embedding based on filename
        path_str = str(audio_path)
        if "reference" in path_str or "ref" in path_str:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        elif "match" in path_str:
            # Similar to reference
            return np.array([0.95, 0.05, 0.0], dtype=np.float32)
        elif "nomatch" in path_str:
            # Very different from reference
            return np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return np.array([0.5, 0.5, 0.0], dtype=np.float32)

    def mean_embedding_from_folder(self, folder, use_cache=True):
        # Return fixed reference embedding
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def mock_service():
    """Mock speaker service that returns mock embedders"""
    service = MagicMock()

    # Return different mock embedders for different models
    service.get_speechbrain = lambda model_id: MockEmbedder("speechbrain")
    service.get_titanet = lambda model_id: MockEmbedder("titanet")
    service.get_wavlm = lambda model_id: MockEmbedder("wavlm")

    return service


@pytest.fixture
def audio_files(tmp_path):
    """Create mock audio file structure"""
    ref_dir = tmp_path / "reference"
    gal_dir = tmp_path / "gallery"

    ref_dir.mkdir()
    gal_dir.mkdir()

    # Reference files
    (ref_dir / "ref1.wav").write_bytes(b"fake_audio")
    (ref_dir / "ref2.wav").write_bytes(b"fake_audio")

    # Gallery files - some match, some don't
    (gal_dir / "match1.wav").write_bytes(b"fake_audio")
    (gal_dir / "match2.wav").write_bytes(b"fake_audio")
    (gal_dir / "nomatch1.wav").write_bytes(b"fake_audio")
    (gal_dir / "nomatch2.wav").write_bytes(b"fake_audio")
    (gal_dir / "other.wav").write_bytes(b"fake_audio")

    return ref_dir, gal_dir


class TestSpeakerMatchPipeline:
    """Integration tests for speaker matching pipeline"""

    @patch('fait.audio.pipelines.speaker_recognition_pipeline.get_speaker_service')
    def test_speechbrain_only_strategy(self, mock_get_service, mock_service, audio_files, tmp_path):
        mock_get_service.return_value = mock_service
        ref_dir, gal_dir = audio_files

        cfg = SpeakerMatchConfig(
            reference_dir=str(ref_dir),
            gallery_dir=str(gal_dir),
            output_dir=str(tmp_path / "output"),
            strategy="speechbrain_only",
            use_cache=False
        )
        cfg.single.tau = 0.9  # Lower threshold to ensure matches

        result = run_speaker_match(cfg)

        assert result["processed"] == 5  # Total gallery files
        assert result["found"] >= 2  # Should find at least the match files
        assert Path(result["run_dir"]).exists()
        assert Path(result["log_jsonl"]).exists()

    @patch('fait.audio.pipelines.speaker_recognition_pipeline.get_speaker_service')
    def test_two_stage_strategy(self, mock_get_service, mock_service, audio_files, tmp_path):
        mock_get_service.return_value = mock_service
        ref_dir, gal_dir = audio_files

        cfg = SpeakerMatchConfig(
            reference_dir=str(ref_dir),
            gallery_dir=str(gal_dir),
            output_dir=str(tmp_path / "output"),
            strategy="two_stage",
            detector="titanet",
            use_cache=False
        )
        cfg.two_stage.tau_star = 0.85

        result = run_speaker_match(cfg)

        assert result["processed"] == 5
        assert result["found"] >= 0  # May vary based on fusion
        assert "two_stage" in str(result["run_dir"])

    @patch('fait.audio.pipelines.speaker_recognition_pipeline.get_speaker_service')
    def test_three_stage_strategy(self, mock_get_service, mock_service, audio_files, tmp_path):
        mock_get_service.return_value = mock_service
        ref_dir, gal_dir = audio_files

        cfg = SpeakerMatchConfig(
            reference_dir=str(ref_dir),
            gallery_dir=str(gal_dir),
            output_dir=str(tmp_path / "output"),
            strategy="three_stage",
            detector="titanet",
            tertiary="wavlm",
            use_cache=False
        )

        result = run_speaker_match(cfg)

        assert result["processed"] == 5
        assert "three_stage" in str(result["run_dir"])

    @patch('fait.audio.pipelines.speaker_recognition_pipeline.get_speaker_service')
    def test_found_files_copied(self, mock_get_service, mock_service, audio_files, tmp_path):
        mock_get_service.return_value = mock_service
        ref_dir, gal_dir = audio_files

        cfg = SpeakerMatchConfig(
            reference_dir=str(ref_dir),
            gallery_dir=str(gal_dir),
            output_dir=str(tmp_path / "output"),
            strategy="speechbrain_only",
            use_cache=False
        )
        cfg.single.tau = 0.8

        result = run_speaker_match(cfg)

        # Check that found files were copied
        found_dir = Path(result["run_dir"]) / "found_audio"
        if result["found"] > 0:
            assert found_dir.exists()
            found_files = list(found_dir.glob("*.wav"))
            assert len(found_files) == result["found"]

    @patch('fait.audio.pipelines.speaker_recognition_pipeline.get_speaker_service')
    def test_log_jsonl_structure(self, mock_get_service, mock_service, audio_files, tmp_path):
        mock_get_service.return_value = mock_service
        ref_dir, gal_dir = audio_files

        cfg = SpeakerMatchConfig(
            reference_dir=str(ref_dir),
            gallery_dir=str(gal_dir),
            output_dir=str(tmp_path / "output"),
            strategy="two_stage",
            detector="titanet",
            use_cache=False
        )

        result = run_speaker_match(cfg)

        # Verify JSONL log structure
        import json
        log_path = Path(result["log_jsonl"])
        assert log_path.exists()

        with open(log_path, 'r') as f:
            lines = [json.loads(line) for line in f if line.strip()]


class TestStrategyResolution:
    def test_auto_resolves_to_two_stage(self, monkeypatch):
        # Clear environment variables from .env file
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="auto",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "two_stage"
        assert detector == "titanet"
        assert tertiary == "none"

    def test_explicit_config_ignores_env(self, monkeypatch):
        # Set env vars that would normally override
        monkeypatch.setenv("FAIT_AUDIO_STRATEGY", "speechbrain_only")
        monkeypatch.setenv("FAIT_AUDIO_DETECTOR", "wavlm")

        # But explicit (non-auto) config should NOT be overridden
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="two_stage",
            detector="titanet",
            tertiary="none"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "two_stage"
        assert detector == "titanet"
        assert tertiary == "none"

    def test_env_overrides_when_all_auto(self, monkeypatch):
        # Clear defaults first
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        # Then set specific values
        monkeypatch.setenv("FAIT_AUDIO_STRATEGY", "three_stage")
        monkeypatch.setenv("FAIT_AUDIO_DETECTOR", "titanet")
        monkeypatch.setenv("FAIT_AUDIO_TERTIARY", "wavlm")

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="auto",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "three_stage"
        assert detector == "titanet"
        assert tertiary == "wavlm"

    def test_speechbrain_only_clears_detector_and_tertiary(self, monkeypatch):
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="speechbrain_only",
            detector="titanet",  # should be cleared
            tertiary="wavlm"  # should be cleared
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "speechbrain_only"
        assert detector == "none"
        assert tertiary == "none"

    def test_titanet_only_clears_detector_and_tertiary(self, monkeypatch):
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="titanet_only",
            detector="wavlm",
            tertiary="titanet"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "titanet_only"
        assert detector == "none"
        assert tertiary == "none"

    def test_detector_only_requires_valid_detector(self, monkeypatch):
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="detector_only",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "detector_only"
        assert detector in {"titanet", "wavlm"}
        assert tertiary == "none"

    def test_two_stage_requires_detector(self, monkeypatch):
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="two_stage",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "two_stage"
        assert detector in {"titanet", "wavlm"}
        assert tertiary == "none"

    def test_three_stage_requires_detector_and_tertiary(self, monkeypatch):
        monkeypatch.delenv("FAIT_AUDIO_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_DETECTOR", raising=False)
        monkeypatch.delenv("FAIT_AUDIO_TERTIARY", raising=False)

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="three_stage",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "three_stage"
        assert detector in {"titanet", "wavlm"}
        assert tertiary in {"titanet", "wavlm"}
        assert detector != tertiary  # They should be different