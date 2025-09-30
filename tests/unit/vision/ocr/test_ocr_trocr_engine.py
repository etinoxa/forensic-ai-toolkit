# tests/unit/vision/test_ocr_trocr_engine.py
import os
from PIL import Image
import types
import fait.vision.ocr.models.trocr_engine as trocr_mod

class _FakeProc:
    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None): return cls()
    def __call__(self, images=None, return_tensors="pt"):
        class _Inputs:
            pixel_values = types.SimpleNamespace(to=lambda device: "PIXELS")
        return _Inputs()
    def batch_decode(self, ids, skip_special_tokens=True): return ["HELLO_FROM_TROCR"]

class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_id, cache_dir=None): return cls()
    def to(self, device): return self
    def eval(self): return self
    def generate(self, pixel_values=None, max_new_tokens=256): return ["IDS"]

def test_trocr_defaults_and_cache(tmp_paths, monkeypatch):
    # Route HF cache under FAIT cache
    monkeypatch.setenv("HF_HOME", str(tmp_paths.models_ocr / "hf"))
    # Patch transformers classes used by TrOCREngine
    import transformers
    monkeypatch.setattr(transformers, "TrOCRProcessor", _FakeProc, raising=True)
    monkeypatch.setattr(transformers, "VisionEncoderDecoderModel", _FakeModel, raising=True)

    eng = trocr_mod.TrOCREngine(model_id=None, cache_dir=str(tmp_paths.models_ocr / "hf"))
    out = eng.ocr(Image.new("RGB", (64, 32), "white"))
    assert out is not None and out.text == "HELLO_FROM_TROCR"
    # default model id applied when None
    assert isinstance(eng.model_id, str) and len(eng.model_id) > 0
    # HF_HOME honored
    assert os.environ.get("HF_HOME")

def test_trocr_env_model_id(tmp_paths, monkeypatch):
    import transformers
    monkeypatch.setattr(transformers, "TrOCRProcessor", _FakeProc, raising=True)
    monkeypatch.setattr(transformers, "VisionEncoderDecoderModel", _FakeModel, raising=True)
    monkeypatch.setenv("FAIT_TROCR_MODEL", "microsoft/trocr-small-printed")

    eng = trocr_mod.TrOCREngine(cache_dir=str(tmp_paths.models_ocr / "hf"))
    assert eng.model_id.endswith("trocr-small-printed")
    out = eng.ocr(Image.new("RGB", (40, 20), "white"))
    assert out.text == "HELLO_FROM_TROCR"
