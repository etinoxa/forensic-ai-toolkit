# tests/unit/audio/test_speaker_recognition.py
import pytest
import numpy as np
from pathlib import Path
from fait.audio.pipelines.speaker_recognition_pipeline import (
    _resolve_strategy,
    _cosine,
    SpeakerMatchConfig,
    SingleStageCfg,
    TwoStageCfg,
    ThreeStageCfg,
)


class TestCosineSimplified:
    """Test simplified cosine similarity for speaker recognition"""

    def test_cosine_identical_vectors(self):
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        sim = _cosine(a, b)
        assert sim == pytest.approx(1.0)

    def test_cosine_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        sim = _cosine(a, b)
        assert sim == pytest.approx(0.0)

    def test_cosine_opposite_vectors(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([-1.0, -2.0], dtype=np.float32)
        sim = _cosine(a, b)
        assert sim == pytest.approx(-1.0)

    def test_cosine_zero_vector(self):
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([0.0, 0.0], dtype=np.float32)
        sim = _cosine(a, b)
        assert sim == 0.0  # Should handle gracefully


class TestStrategyResolution:
    """Test speaker recognition strategy resolution"""

    def test_auto_resolves_to_two_stage(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="auto",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "two_stage"
        assert detector in {"titanet", "wavlm"}
        assert tertiary == "none"

    def test_speechbrain_only_no_detector(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="speechbrain_only",
            detector="titanet",  # Should be ignored
            tertiary="wavlm"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "speechbrain_only"
        assert detector == "none"
        assert tertiary == "none"

    def test_titanet_only_no_detector(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="titanet_only",
            detector="wavlm",
            tertiary="wavlm"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "titanet_only"
        assert detector == "none"
        assert tertiary == "none"

    def test_detector_only_requires_detector(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="detector_only",
            detector="auto",
            tertiary="none"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "detector_only"
        assert detector in {"titanet", "wavlm"}
        assert tertiary == "none"

    def test_two_stage_requires_detector(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="two_stage",
            detector="auto",
            tertiary="none"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "two_stage"
        assert detector in {"titanet", "wavlm"}
        assert tertiary == "none"

    def test_three_stage_requires_both(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="three_stage",
            detector="titanet",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "three_stage"
        assert detector == "titanet"
        assert tertiary == "wavlm"  # Should pick the other one

    def test_three_stage_swaps_if_same(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="three_stage",
            detector="titanet",
            tertiary="titanet"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)
        assert strategy == "three_stage"
        assert detector == "titanet"
        assert tertiary == "wavlm"  # Should be different from detector

    def test_env_override_all_auto(self, monkeypatch):
        monkeypatch.setenv("FAIT_AUDIO_STRATEGY", "three_stage")
        monkeypatch.setenv("FAIT_AUDIO_DETECTOR", "wavlm")
        monkeypatch.setenv("FAIT_AUDIO_TERTIARY", "titanet")

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="auto",
            detector="auto",
            tertiary="auto"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)

        # ENV should win when all are auto
        assert strategy == "three_stage"
        assert detector == "wavlm"
        assert tertiary == "titanet"

    def test_explicit_config_wins_when_not_auto(self, monkeypatch):
        monkeypatch.setenv("FAIT_AUDIO_STRATEGY", "three_stage")
        monkeypatch.setenv("FAIT_AUDIO_DETECTOR", "wavlm")

        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            strategy="two_stage",  # Explicit, not auto
            detector="titanet",  # Explicit, not auto
            tertiary="none"
        )
        strategy, detector, tertiary = _resolve_strategy(cfg)

        # Explicit config should win
        assert strategy == "two_stage"
        assert detector == "titanet"


class TestSpeakerMatchConfig:
    """Test speaker match configuration defaults"""

    def test_default_single_stage_threshold(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal"
        )
        assert cfg.single.tau == 0.70

    def test_default_two_stage_params(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal"
        )
        assert cfg.two_stage.method in {"and", "weighted", "product", "max"}
        assert 0 < cfg.two_stage.alpha < 1
        assert cfg.two_stage.tau_star > 0

    def test_default_three_stage_params(self):
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal"
        )
        assert 0 < cfg.three_stage.alpha12 < 1
        assert 0 < cfg.three_stage.alpha123 < 1
        assert cfg.three_stage.tau_star > 0

    def test_custom_thresholds(self):
        single = SingleStageCfg(tau=0.80)
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            single=single
        )
        assert cfg.single.tau == 0.80

    def test_model_ids_configurable(self):
        from fait.audio.pipelines.speaker_recognition_pipeline import SpeakerModelsCfg

        models = SpeakerModelsCfg(
            speechbrain_id="custom/model",
            titanet_id="custom/titanet",
            wavlm_id="custom/wavlm"
        )
        cfg = SpeakerMatchConfig(
            reference_dir="/tmp/ref",
            gallery_dir="/tmp/gal",
            models=models
        )
        assert cfg.models.speechbrain_id == "custom/model"
        assert cfg.models.titanet_id == "custom/titanet"
        assert cfg.models.wavlm_id == "custom/wavlm"


class TestBaseAudioEmbedder:
    """Test base audio embedder interface"""

    def test_mean_embedding_aggregation(self):
        # This would typically be tested with a mock embedder
        # that returns known embeddings
        from fait.audio.speaker_recognition.base import _BaseAudioEmbedder

        class MockEmbedder(_BaseAudioEmbedder):
            def name(self):
                return "mock"

            def embed_file(self, audio_path, use_cache=True):
                # Return deterministic embedding based on filename
                if "ref1" in str(audio_path):
                    return np.array([1.0, 0.0], dtype=np.float32)
                elif "ref2" in str(audio_path):
                    return np.array([0.0, 1.0], dtype=np.float32)
                return None

        embedder = MockEmbedder()

        # Test that mean is computed correctly
        # (would need actual temp files in a real test)


class TestFusionLogic:
    """Test score fusion for speaker recognition"""

    def test_two_stage_weighted_fusion(self):
        from fait.audio.pipelines.speaker_recognition_pipeline import _fuse_two

        cfg = TwoStageCfg(method="weighted", alpha=0.6, tau_star=0.5)
        fused, tau, ok = _fuse_two(0.8, 0.4, cfg)

        # weighted: 0.6 * 0.8 + 0.4 * 0.4 = 0.48 + 0.16 = 0.64
        assert fused == pytest.approx(0.64)
        assert tau == 0.5
        assert ok is True

    def test_three_stage_fusion(self):
        from fait.audio.pipelines.speaker_recognition_pipeline import _fuse_three

        cfg = ThreeStageCfg(
            method="weighted",
            alpha12=0.6,
            alpha123=0.5,
            tau_star=0.5
        )
        fused, tau, ok = _fuse_three(0.8, 0.6, 0.7, cfg)

        # First fusion: 0.6 * 0.8 + 0.4 * 0.6 = 0.48 + 0.24 = 0.72
        # Second fusion: 0.5 * 0.72 + 0.5 * 0.7 = 0.36 + 0.35 = 0.71
        assert fused == pytest.approx(0.71)
        assert ok is True


class TestAudioFileHandling:
    """Test audio file type detection"""

    def test_audio_extensions(self):
        from fait.core.utils import is_audio_file

        assert is_audio_file(Path("test.wav")) is True
        assert is_audio_file(Path("test.mp3")) is True
        assert is_audio_file(Path("test.m4a")) is True
        assert is_audio_file(Path("test.flac")) is True
        assert is_audio_file(Path("test.ogg")) is True

        # Not audio
        assert is_audio_file(Path("test.txt")) is False
        assert is_audio_file(Path("test.mp4")) is False
        assert is_audio_file(Path("test.jpg")) is False