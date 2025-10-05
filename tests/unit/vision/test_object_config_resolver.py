from fait.vision.pipelines.object_screen import ScreenConfig, _resolve_strategy_verifier

def test_auto_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "detector_only")
    monkeypatch.setenv("FAIT_OBJECT_VERIFIER", "yolo")
    cfg = ScreenConfig(strategy="auto", verifier="auto", prompts=["knife"], gallery_dir=str(tmp_path))
    s, v = _resolve_strategy_verifier(cfg)
    assert (s, v) == ("detector_only", "yolo")
