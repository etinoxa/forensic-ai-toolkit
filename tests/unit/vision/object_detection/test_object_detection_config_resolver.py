# tests/unit/vision/test_object_detection_config.py

from fait.vision.pipelines.object_detection_pipeline import ScreenConfig, _resolve_strategy_verifier


class TestObjectStrategyResolution:
    def test_auto_uses_config_yaml(self, monkeypatch, tmp_path):
        """When auto and no env vars, should load from config.yaml"""
        monkeypatch.delenv("FAIT_OBJECT_STRATEGY", raising=False)
        monkeypatch.delenv("FAIT_OBJECT_VERIFIER", raising=False)

        # Create test config
        test_config = tmp_path / "test_config.yaml"
        test_config.write_text("""
vision:
  object_detection:
    strategy: two_stage
    verifier: yolo
""")

        # Mock config loader
        from fait.core import app_config
        monkeypatch.setattr(app_config, "_app_config", None)
        original_load = app_config.FaitConfig.load
        monkeypatch.setattr(
            app_config.FaitConfig,
            "load",
            lambda path=None: original_load(test_config)
        )

        cfg = ScreenConfig(
            prompts=["knife"],
            gallery_dir="/tmp/gal",
            strategy="auto",
            verifier="auto"
        )
        strategy, verifier = _resolve_strategy_verifier(cfg)
        assert strategy == "two_stage"
        assert verifier == "yolo"

    def test_env_overrides_auto(self, monkeypatch):
        """Environment overrides when both are auto"""
        monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "gdino_only")
        monkeypatch.setenv("FAIT_OBJECT_VERIFIER", "none")

        cfg = ScreenConfig(
            prompts=["knife"],
            gallery_dir="/tmp/gal",
            strategy="auto",
            verifier="auto"
        )
        strategy, verifier = _resolve_strategy_verifier(cfg)
        assert strategy == "gdino_only"
        assert verifier == "none"

    def test_explicit_ignores_env(self, monkeypatch):
        """Explicit values not overridden by environment"""
        monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "gdino_only")
        monkeypatch.setenv("FAIT_OBJECT_VERIFIER", "none")

        cfg = ScreenConfig(
            prompts=["knife"],
            gallery_dir="/tmp/gal",
            strategy="detector_only",
            verifier="yolo"
        )
        strategy, verifier = _resolve_strategy_verifier(cfg)
        assert strategy == "detector_only"
        assert verifier == "yolo"


