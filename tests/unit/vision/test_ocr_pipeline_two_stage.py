# tests/unit/vision/test_ocr_pipeline_two_stage.py
import types
from PIL import Image
import fait.vision.pipelines.ocr_pipeline as ocrmod
import fait.vision.ocr.engines as engreg

def test_two_stage_prefers_verifier(monkeypatch, tmp_paths, tmp_path):
    monkeypatch.setattr(ocrmod, "get_paths", lambda: tmp_paths, raising=False)

    gal = tmp_path / "gal"; gal.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 32), "white").save(gal / "a.png")

    class _FakePaddlePrimary:
        def ocr(self, img, lang="auto"):
            # primary returns weaker text
            return "PRIMARY", 0.5

    class _FakeVerifier:
        def ocr(self, img, lang="auto"):
            return "VERIFIED_TEXT", 0.99

    orig_get = engreg.get_engine
    def _fake_get_engine(name, cfg=None):
        if name == "paddle": return _FakePaddlePrimary()
        if name == "trocr":  return _FakeVerifier()
        return orig_get(name, cfg)
    monkeypatch.setattr(engreg, "get_engine", _fake_get_engine, raising=True)

    class Cfg: pass
    cfg = Cfg()
    cfg.strategy = "two_stage"
    cfg.verifier = "trocr"
    cfg.gallery_dir = str(gal)
    cfg.output_dir = str(tmp_paths.outputs)
    cfg.rotations = [0]
    cfg.exts = [".png"]
    cfg.min_file_kb = 0
    cfg.min_dim_px = [0, 0]
    cfg.engines = {"paddle": {"enabled": True}, "trocr": {"enabled": True}}

    out = ocrmod.run_ocr(cfg)
    assert out["processed"] == 1
    assert out["found"] >= 1
    assert "two_stage" in out["run_dir"]
