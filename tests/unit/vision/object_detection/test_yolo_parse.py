# tests/unit/vision/test_yolo_parse.py
import pytest
pytestmark = pytest.mark.requires_models
import torch
import fait.vision.object_detection.models.yolo as y
from fait.vision.object_detection.models.yolo import YOLODetector, YoloConfig

def _get(v, k):
    return getattr(v, k) if hasattr(v, k) else v[k]

class DummyBox:
    """One detection entry with shapes matching Ultralytics conventions."""
    def __init__(self):
        # 1 detection → xyxy is (1,4), conf and cls are (1,)
        self.xyxy = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        self.conf = torch.tensor([0.9], dtype=torch.float32)
        self.cls  = torch.tensor([0.0], dtype=torch.float32)  # class id 0

class DummyResult:
    def __init__(self):
        # Your detector does: for b in res.boxes: ...
        # So make boxes iterable by being a list of DummyBox objects.
        self.boxes = [DummyBox()]

class DummyModel:
    names = {0: "knife"}  # mapping used by detector
    def predict(self, **_): return [DummyResult()]
    def to(self, *_a, **_k): return self

def test_yolo_detect_parse(monkeypatch, tmp_paths):
    # Route module to tmp paths and use our dummy model
    monkeypatch.setattr(y, "get_paths", lambda: tmp_paths)
    monkeypatch.setattr(y, "YOLO", lambda *_: DummyModel())

    # Provide a fake weights file where resolver expects it
    weights = tmp_paths.models_object_detection / "local.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.write_bytes(b"x")
    monkeypatch.setattr(y, "_resolve_yolo_weights", lambda _id, _base: weights)

    det = YOLODetector(
        YoloConfig(model_id="local.pt", class_whitelist=["knife"]),
        cache_dir=str(tmp_paths.models_object_detection),
    )

    out = det.detect("anything.jpg")

    # Expect a list[Detection]
    d = out[0]


    assert _get(d, "label") == "knife"
    assert list(_get(d, "bbox")) == [1.0, 2.0, 3.0, 4.0]
    assert 0.89 < _get(d, "score") < 0.91