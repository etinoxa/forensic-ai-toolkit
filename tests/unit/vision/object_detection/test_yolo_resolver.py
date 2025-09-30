# tests/unit/vision/test_yolo_resolver.py
from fait.vision.object_detection.models.yolo import _resolve_yolo_weights

def test_yolo_resolver_local(tmp_paths, tmp_path):
    local = tmp_path/"yolov8n.pt"; local.write_bytes(b"x")
    assert _resolve_yolo_weights(str(local), tmp_paths.models_object_detection) == local.resolve()
