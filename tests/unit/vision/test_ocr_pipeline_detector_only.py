# tests/unit/vision/test_ocr_pipeline_detector_only.py
import types
from PIL import Image
import fait.vision.pipelines.ocr_pipeline as ocrmod
import fait.vision.ocr.engines as engreg

def test_detector_only_calls_paddle_and_verifier(monkeypatch, tmp_paths, tmp_path):
    # Route FAIT paths to tmp
    monkeypatch.setattr(ocrmod, "get_paths", lambda: tmp_paths, raising=False)

    # Build a tiny gallery
    gal = tmp_path / "gal"; gal.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 32), "white").save(gal / "a.png")
    Image.new("RGB", (64, 32), "white").save(gal / "b.png")

    # Fake Paddle detector engine
    class _FakePaddleDet:
        def detect(self, img):
            # return two crops
            return [([[1,1],[30,1],[30,16],[1,16]], Image.new("RGB",(29,15),"white")),
                    ([[2,2],[40,2],[40,18],[2,18]], Image.new("RGB",(38,16),"white"))]

    # Fake verifier (e.g., trocr/doctr/tesseract wrapper) that returns text
    class _FakeVerifier:
        def recognize(self, img):
            return types.SimpleNamespace(text="OK", confidence=0.8)

    # Patch engine registry so _eng("paddle") → detector, _eng("trocr") → verifier
    orig_get = engreg.get_engine
    def _fake_get_engine(name, cfg=None):
        if name == "paddle": return _FakePaddleDet()
        if name == "trocr":  return _FakeVerifier()
        return orig_get(name, cfg)
    monkeypatch.setattr(engreg, "get_engine", _fake_get_engine, raising=True)

    # Minimal cfg object matching pipeline expectations
    class Cfg: pass
    cfg = Cfg()
    cfg.strategy = "detector_only"
    cfg.verifier = "trocr"
    cfg.gallery_dir = str(gal)
    cfg.output_dir = str(tmp_paths.outputs)
    cfg.rotations = [0]
    cfg.exts = [".png"]
    cfg.min_file_kb = 0
    cfg.min_dim_px = [0, 0]
    cfg.engines = {"paddle": {"enabled": True}, "trocr": {"enabled": True}}
    cfg.fusion = ocrmod.FusionCfg()
    cfg.engine_order = ["paddle", "trocr", "tesseract"]

    out = ocrmod.run_ocr(cfg)
    assert out["processed"] == 2
    assert out["found"] >= 1
    assert "detector_only" in out["run_dir"]
