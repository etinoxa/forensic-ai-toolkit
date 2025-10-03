# tests/unit/vision/test_ocr_config.py
import pytest
from fait.core.utils import resolve_strategy_verifier, resolve_engine_order
from fait.vision.ocr.models.config import OcrConfig, FusionCfg


class TestStrategyVerifierResolution:
    """Test OCR strategy and verifier resolution logic"""

    def test_two_stage_assigns_default_verifier(self, monkeypatch):
        # Ensure clean environment
        monkeypatch.delenv("FAIT_OCR_STRATEGY", raising=False)
        monkeypatch.delenv("OCR_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_OCR_VERIFIER", raising=False)
        monkeypatch.delenv("OCR_VERIFIER", raising=False)

        fusion = FusionCfg(strategy="two_stage", verifier="none")
        strategy, verifier = resolve_strategy_verifier(fusion)
        assert strategy == "two_stage"
        assert verifier == "tesseract"

    def test_detector_only_assigns_default_verifier(self, monkeypatch):
        # Ensure clean environment
        monkeypatch.delenv("FAIT_OCR_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_OCR_VERIFIER", raising=False)

        fusion = FusionCfg(strategy="detector_only", verifier="none")
        strategy, verifier = resolve_strategy_verifier(fusion)
        assert strategy == "detector_only"
        assert verifier == "paddle"

    def test_env_overrides_auto_strategy(self, monkeypatch):
        # Test that env vars work when strategy/verifier are "auto"
        monkeypatch.setenv("FAIT_OCR_STRATEGY", "two_stage")
        monkeypatch.setenv("FAIT_OCR_VERIFIER", "tesseract")

        fusion = FusionCfg(strategy="auto", verifier="auto")
        strategy, verifier = resolve_strategy_verifier(fusion)
        assert strategy == "two_stage"
        assert verifier == "tesseract"

    def test_explicit_config_ignores_env(self, monkeypatch):
        # Test that explicit config values are NOT overridden by env
        monkeypatch.setenv("FAIT_OCR_STRATEGY", "first_nonempty")
        monkeypatch.setenv("FAIT_OCR_VERIFIER", "none")

        fusion = FusionCfg(strategy="two_stage", verifier="tesseract")
        strategy, verifier = resolve_strategy_verifier(fusion)
        assert strategy == "two_stage"
        assert verifier == "tesseract"


class TestEngineOrderResolution:
    """Test OCR engine order resolution"""

    def test_filters_disabled_engines(self):
        cfg = OcrConfig(
            engines={
                "tesseract": {"enabled": True},
                "trocr": {"enabled": False},
                "paddle": {"enabled": True},
            },
            engine_order=["tesseract", "trocr", "paddle"]
        )

        order = resolve_engine_order(cfg, "first_nonempty")
        assert order == ["tesseract", "paddle"]
        assert "trocr" not in order

    def test_preserves_order(self):
        cfg = OcrConfig(
            engines={
                "tesseract": {"enabled": True},
                "trocr": {"enabled": True},
                "paddle": {"enabled": True},
            },
            engine_order=["paddle", "tesseract", "trocr"]
        )

        order = resolve_engine_order(cfg, "first_nonempty")
        assert order == ["paddle", "tesseract", "trocr"]

    def test_empty_when_all_disabled(self):
        cfg = OcrConfig(
            engines={
                "tesseract": {"enabled": False},
                "trocr": {"enabled": False},
            },
            engine_order=["tesseract", "trocr"]
        )

        order = resolve_engine_order(cfg, "first_nonempty")
        assert order == []

    def test_env_override_for_first_nonempty(self, monkeypatch):
        monkeypatch.setenv("FAIT_OCR_ENGINES", "trocr,tesseract")

        cfg = OcrConfig(
            engines={
                "tesseract": {"enabled": True},
                "trocr": {"enabled": True},
                "paddle": {"enabled": True},
            },
            engine_order=["paddle", "tesseract"]
        )

        order = resolve_engine_order(cfg, "first_nonempty")
        # ENV should override for first_nonempty strategy
        assert order == ["trocr", "tesseract"]

    def test_no_env_override_for_two_stage(self, monkeypatch):
        monkeypatch.setenv("FAIT_OCR_ENGINES", "trocr,tesseract")

        cfg = OcrConfig(
            engines={
                "paddle": {"enabled": True},
                "trocr": {"enabled": True},
            },
            engine_order=["paddle"]
        )

        order = resolve_engine_order(cfg, "two_stage")
        # ENV should NOT override for two_stage
        assert order == ["paddle"]


class TestOcrConfigDefaults:
    """Test OCR config default values"""

    def test_default_config(self):
        cfg = OcrConfig()
        assert cfg.min_file_kb == 10
        assert cfg.min_dim_px == 100
        assert 0 in cfg.rotations

    def test_fusion_defaults(self):
        fusion = FusionCfg()
        assert fusion.strategy in {"first_nonempty", "best_of", "consensus", "two_stage", "auto"}
        assert fusion.alpha > 0 and fusion.alpha < 1
        assert fusion.sim_tau > 0 and fusion.sim_tau <= 1