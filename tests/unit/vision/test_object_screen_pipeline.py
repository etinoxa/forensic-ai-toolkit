# tests/unit/vision/test_object_screen_pipeline.py
import io, json
from PIL import Image
from types import SimpleNamespace
from pathlib import Path
import fait.core.paths as paths_mod
import fait.vision.pipelines.object_screen as objmod
from fait.vision.detectors.base import Detection

def _mk_png(path):
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)

def test_object_screen_detector_only(monkeypatch, tmp_paths, tmp_path):
    # Route object_screen paths to tmp
    monkeypatch.setattr(paths_mod, "get_paths", lambda: tmp_paths, raising=False)

    # Force strategy/verifier
    monkeypatch.setattr(objmod, "_resolve_strategy_verifier",
                        lambda cfg: ("detector_only", "yolo"))

    # Stub detector returned by the service
    class StubDet:
        def detect(self, img_or_path):
            return [Detection(bbox=[1,2,10,12], score=0.92, label="knife")]

    # Stub service to return our detector
    import fait.vision.services.object_service as svcmod
    class StubSvc:
        def get_yolo(self, *_a, **_k): return StubDet()
        def get_gdino(self, *_a, **_k): raise AssertionError("Not called")
        def get_detr(self, *_a, **_k): raise AssertionError("Not called")
    monkeypatch.setattr(svcmod, "ObjectModelsService", lambda *_: StubSvc())

    # Build a tiny gallery
    gal = tmp_path / "gal"
    gal.mkdir(parents=True, exist_ok=True)
    _mk_png(gal / "a.png")
    _mk_png(gal / "b.png")

    # Config: keep it simple
    cfg = objmod.ScreenConfig(
        gallery_dir=str(gal),
        prompts=["knife"],          # ignored by detector_only; safe to set
    )
    # Lower thresholds so we certainly accept
    cfg.detector_only.default_tau = 0.5
    cfg.detector_only.class_thresholds = {"knife": 0.5}

    summary = objmod.run_object_screen(cfg)

    # Assertions
    assert summary["processed"] == 2
    assert summary["found"] == 2
    assert (gal.exists())
    # report file created
    assert summary["report_path"]
    run_dir = Path(summary["run_dir"]).resolve()
    fait_root = tmp_paths.fait_root.resolve()
    outputs_root = tmp_paths.outputs.resolve()

    assert run_dir.exists() and run_dir.is_dir()

    # JSONL lines exist
    log_path = summary.get("log_jsonl")

    if not log_path:
        log_path = Path(summary["run_dir"]) / "log.jsonl"
    else:
        log_path = Path(log_path)
    assert log_path.exists()
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    assert any(e.get("detector_label") == "knife" for e in lines)
