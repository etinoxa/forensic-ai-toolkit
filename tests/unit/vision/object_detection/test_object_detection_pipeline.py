# tests/unit/vision/test_object_detection_pipeline.py
import sys
import json
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from PIL import Image
from pathlib import Path
import fait.core.paths as paths_mod
import fait.vision.pipelines.object_detection_pipeline as objmod
from fait.vision.object_detection.base import Detection


def _mk_png(path):
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)


def test_object_detection_detector_only(monkeypatch, tmp_paths, tmp_path):
    # Route paths to tmp
    monkeypatch.setattr(paths_mod, "get_paths", lambda: tmp_paths, raising=False)

    # Force strategy/verifier
    monkeypatch.setattr(objmod, "_resolve_strategy_verifier",
                        lambda cfg: ("detector_only", "yolo"))

    # Stub detector
    class StubDet:
        def detect(self, img_or_path):
            return [Detection(bbox=[1, 2, 10, 12], score=0.92, label="knife")]

    # Stub service
    class StubSvc:
        def get_yolo(self, *_a, **_k):
            return StubDet()

        def get_gdino(self, *_a, **_k):
            raise AssertionError("Not called")

        def get_detr(self, *_a, **_k):
            raise AssertionError("Not called")

    # Complete stub config classes (with all attributes the pipeline logs)
    @dataclass
    class FakeGDINOConfig:
        model_id: str = "fake-gdino"
        box_threshold: float = 0.3
        text_threshold: float = 0.3
        nms_iou: float = 0.5
        long_side: int = 1024
        box_expand: float = 0.2
        normalize_prompts: bool = True
        add_relational_prompts: bool = True

    @dataclass
    class FakeDefDETRConfig:
        model_id: str = "fake-detr"
        score_threshold: float = 0.25
        nms_iou: float = 0.5
        class_whitelist: Optional[List[str]] = None

    @dataclass
    class FakeYoloConfig:
        model_id: str = "fake-yolo"
        score_threshold: float = 0.25
        nms_iou: float = 0.5
        imgsz: int = 640
        class_whitelist: Optional[List[str]] = None

    # Create fake modules
    fake_svc_module = SimpleNamespace(
        get_object_service=lambda: StubSvc(),
        ObjectModelsService=StubSvc
    )
    fake_gdino_module = SimpleNamespace(GDINOConfig=FakeGDINOConfig)
    fake_detr_module = SimpleNamespace(DefDETRConfig=FakeDefDETRConfig)
    fake_yolo_module = SimpleNamespace(YoloConfig=FakeYoloConfig)

    # Inject into sys.modules
    monkeypatch.setitem(sys.modules, 'fait.vision.services.object_detection_service', fake_svc_module)
    monkeypatch.setitem(sys.modules, 'fait.vision.object_detection.models.grounding_dino', fake_gdino_module)
    monkeypatch.setitem(sys.modules, 'fait.vision.object_detection.models.deformable_detr', fake_detr_module)
    monkeypatch.setitem(sys.modules, 'fait.vision.object_detection.models.yolo', fake_yolo_module)

    # Build gallery
    gal = tmp_path / "gal"
    gal.mkdir(parents=True, exist_ok=True)
    _mk_png(gal / "a.png")
    _mk_png(gal / "b.png")

    # Config
    cfg = objmod.ScreenConfig(
        gallery_dir=str(gal),
        prompts=["knife"],
    )
    cfg.detector_only.default_tau = 0.5
    cfg.detector_only.class_thresholds = {"knife": 0.5}

    summary = objmod.run_object_detection(cfg)

    # Assertions
    assert summary["processed"] == 2
    assert summary["found"] == 2
    assert Path(summary["run_dir"]).exists()
    assert summary.get("report_path")

    # Check JSONL
    run_dir = Path(summary["run_dir"])
    log_path = run_dir / "log.jsonl"
    assert log_path.exists()

    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 2
    assert any(e.get("detector_label") == "knife" for e in lines)