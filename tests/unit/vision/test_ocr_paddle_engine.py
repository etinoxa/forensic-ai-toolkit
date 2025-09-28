# tests/unit/vision/test_ocr_paddle_engine.py
import os, types, numpy as np
from PIL import Image
import fait.vision.ocr.engines.paddle_engine as paddle_mod

class _FakePaddleOCR:
    def __init__(self, *a, **k): pass

    def ocr(self, img, det=True, rec=False, cls=False):
        # Return ONE page (outer list) containing TWO items (inner list)
        if det and not rec:
            return [[
                {"points": [[1, 1], [60, 1], [60, 25], [1, 25]]},
                ([[2, 2], [30, 2], [30, 18], [2, 18]],)
            ]]
        # If recognition path is used by mistake, still return something harmless
        return [[([[1, 1], [60, 1], [60, 25], [1, 25]], ("TEXT", 0.9))]]

def test_paddle_detect_contract(monkeypatch, tmp_paths):
    # Force PaddleOCR stub
    import paddleocr
    monkeypatch.setattr(paddleocr, "PaddleOCR", _FakePaddleOCR, raising=True)

    eng = paddle_mod.PaddleEngine(lang="en", cache_dir=str(tmp_paths.models_cache / "ocr" / ".paddlex"))
    boxes = eng.detect(Image.new("RGB", (80, 40), "white"))
    assert isinstance(boxes, list) and len(boxes) >= 2
    poly, crop = boxes[0]
    assert hasattr(crop, "size") and crop.size[0] > 0 and crop.size[1] > 0
    # Envs for cache should be set
    assert "PADDLEX_HOME" in os.environ or "PADDLEOCR_HOME" in os.environ
