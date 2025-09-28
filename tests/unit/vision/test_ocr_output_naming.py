# tests/unit/vision/test_ocr_output_naming.py
from PIL import Image
import fait.vision.pipelines.ocr_pipeline as ocrmod

def _mk_cfg(tmp_paths, gal, strategy, verifier):
    class Cfg: pass
    cfg = Cfg()
    cfg.strategy = strategy
    cfg.verifier = verifier
    cfg.gallery_dir = str(gal)
    cfg.output_dir = str(tmp_paths.outputs)
    cfg.rotations = [0]
    cfg.exts = [".png"]
    cfg.min_file_kb = 0
    cfg.min_dim_px = [0, 0]
    # engines presence just to satisfy pipeline guards
    cfg.engines = {"paddle": {"enabled": True}, "trocr": {"enabled": True}, "tesseract": {"enabled": True}}
    return cfg

def test_output_naming_contains_strategy(monkeypatch, tmp_paths, tmp_path):
    monkeypatch.setattr(ocrmod, "get_paths", lambda: tmp_paths, raising=False)
    gal = tmp_path / "gal"; gal.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), "white").save(gal / "a.png")

    # Detector only
    cfg = _mk_cfg(tmp_paths, gal, "detector_only", "tesseract")
    out = ocrmod.run_ocr(cfg)
    assert "detector_only" in out["run_dir"]

    # Two-stage
    cfg2 = _mk_cfg(tmp_paths, gal, "two_stage", "trocr")
    out2 = ocrmod.run_ocr(cfg2)
    assert "two_stage" in out2["run_dir"]
