# tests/unit/core/test_config_resolver.py
import os
from fait.vision.pipelines.object_screen import ScreenConfig, _resolve_strategy_verifier

def test_auto_env_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "detector_only")
    monkeypatch.setenv("FAIT_OBJECT_VERIFIER", "yolo")

    cfg = ScreenConfig(
        strategy="auto",
        verifier="auto",
        prompts=["knife"],               # <- minimal
        gallery_dir=str(tmp_path),       # <- minimal
    )
    s, v = _resolve_strategy_verifier(cfg)
    assert (s, v) == ("detector_only", "yolo")
