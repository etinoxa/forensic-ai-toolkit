# tests/unit/core/test_config_resolver.py
import sys
import types
from unittest.mock import MagicMock

# Mock the heavy imports BEFORE they're attempted
sys.modules['fait.vision.services.object_detection_service'] = types.ModuleType('fait.vision.services.object_detection_service')
sys.modules['fait.vision.object_detection.models.deformable_detr'] = types.ModuleType('fait.vision.object_detection.models.deformable_detr')
sys.modules['fait.vision.object_detection.models.grounding_dino'] = types.ModuleType('fait.vision.object_detection.models.grounding_dino')
sys.modules['fait.vision.object_detection.models.yolo'] = types.ModuleType('fait.vision.object_detection.models.yolo')

# Now we can import the resolver function
from fait.vision.pipelines.object_detection_pipeline import ScreenConfig, _resolve_strategy_verifier

def test_auto_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "detector_only")
    monkeypatch.setenv("FAIT_OBJECT_VERIFIER", "yolo")
    cfg = ScreenConfig(strategy="auto", verifier="auto", prompts=["knife"], gallery_dir=str(tmp_path))
    s, v = _resolve_strategy_verifier(cfg)
    assert (s, v) == ("detector_only", "yolo")