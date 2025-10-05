# tests/unit/vision/object_detection/test_deformable_detr_parse.py
import pytest
pytestmark = pytest.mark.requires_models
from types import SimpleNamespace
import torch
import pytest
from PIL import Image
import sys
from pathlib import Path

# Add src to path if not already there
src_path = Path(__file__).resolve().parents[4] / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import fait.core.paths as paths_mod
# Use direct file import instead of package import
from fait.vision.object_detection.models import deformable_detr as detrmod
from fait.vision.object_detection.base import Detection

def test_detr_detect_minimal(monkeypatch, tmp_paths):
    monkeypatch.setattr(paths_mod, "get_paths", lambda: tmp_paths, raising=False)

    # Fake processor
    class FakeProc:
        def __call__(self, images, return_tensors="pt"):
            return {"pixel_values": torch.zeros(1,3,16,16)}
        def post_process_object_detection(self, outputs, threshold, target_sizes):
            return [{
                "boxes": torch.tensor([[1.,2.,3.,4.]]),
                "scores": torch.tensor([0.88]),
                "labels": torch.tensor([0]),
            }]

    # Fake model
    class FakeModel:
        def __init__(self): self.config = SimpleNamespace(id2label={0: "knife"})
        def to(self, *_a, **_k): return self
        def eval(self): return self
        def __call__(self, **_): return SimpleNamespace()

    # Bypass __init__ heavy work
    def fake_init(self, cfg, cache_dir=None):
        self.cfg = SimpleNamespace(score_threshold=0.1, class_whitelist=None)
        self.device = "cpu"
        self.processor = FakeProc()
        self.model = FakeModel()
    monkeypatch.setattr(detrmod.DeformableDETR, "__init__", fake_init)

    det = detrmod.DeformableDETR(cfg=SimpleNamespace())
    img = Image.new("RGB", (20, 10))
    out = det.detect(img)
    assert isinstance(out, list) and isinstance(out[0], Detection)
    d = out[0]
    assert d.label == "knife"
    assert d.score == pytest.approx(0.88, rel=1e-6)
    assert d.bbox == [1.0, 2.0, 3.0, 4.0]