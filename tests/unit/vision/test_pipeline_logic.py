# tests/unit/vision/test_pipeline_logic.py
import pytest
from pathlib import Path
from fait.vision.pipelines.object_detection_pipeline import (
    _resolve_strategy_verifier,
    _iou_xyxy,
    _pick_class_threshold,
    ScreenConfig,
    FusionConfig,
)


class TestIoUCalculation:
    """Test Intersection over Union calculations"""

    def test_iou_perfect_overlap(self):
        box_a = [0, 0, 10, 10]
        box_b = [0, 0, 10, 10]
        iou = _iou_xyxy(box_a, box_b)
        assert iou == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 30, 30]
        iou = _iou_xyxy(box_a, box_b)
        assert iou == pytest.approx(0.0)

    def test_iou_partial_overlap(self):
        box_a = [0, 0, 10, 10]  # area = 100
        box_b = [5, 5, 15, 15]  # area = 100
        # intersection = 5x5 = 25
        # union = 100 + 100 - 25 = 175
        # iou = 25/175 ≈ 0.143
        iou = _iou_xyxy(box_a, box_b)
        assert iou == pytest.approx(25 / 175, rel=1e-3)

    def test_iou_contained(self):
        box_a = [0, 0, 20, 20]  # area = 400
        box_b = [5, 5, 15, 15]  # area = 100 (inside box_a)
        # intersection = 100
        # union = 400
        # iou = 100/400 = 0.25
        iou = _iou_xyxy(box_a, box_b)
        assert iou == pytest.approx(0.25)


class TestThresholdSelection:
    """Test class-specific threshold selection"""

    def test_pick_class_threshold_exact_match(self):
        fcfg = FusionConfig(
            class_thresholds={"knife": 0.45, "gun": 0.40},
            gdino_only_default_tau=0.50
        )
        tau = _pick_class_threshold(fcfg, "knife")
        assert tau == 0.45

    def test_pick_class_threshold_case_insensitive(self):
        fcfg = FusionConfig(
            class_thresholds={"knife": 0.45},
            gdino_only_default_tau=0.50
        )
        tau = _pick_class_threshold(fcfg, "KNIFE")
        assert tau == 0.45

    def test_pick_class_threshold_default(self):
        fcfg = FusionConfig(
            class_thresholds={"knife": 0.45},
            gdino_only_default_tau=0.50
        )
        tau = _pick_class_threshold(fcfg, "unknown_class")
        assert tau == 0.50


class TestStrategyVerifierResolution:
    """Test object detection strategy and verifier resolution"""

    def test_auto_defaults_to_two_stage(self):
        cfg = ScreenConfig(strategy="auto", verifier="auto", prompts=["knife"], gallery_dir="/tmp")
        s, v = _resolve_strategy_verifier(cfg)
        assert s == "two_stage"
        assert v in {"yolo", "deformable_detr"}

    def test_gdino_only_forces_no_verifier(self):
        cfg = ScreenConfig(strategy="gdino_only", verifier="yolo", prompts=["knife"], gallery_dir="/tmp")
        s, v = _resolve_strategy_verifier(cfg)
        assert s == "gdino_only"
        assert v == "none"

    def test_detector_only_requires_verifier(self):
        cfg = ScreenConfig(strategy="detector_only", verifier="none", prompts=["knife"], gallery_dir="/tmp")
        s, v = _resolve_strategy_verifier(cfg)
        assert s == "detector_only"
        assert v in {"yolo", "deformable_detr"}

    def test_env_overrides_auto(self, monkeypatch):
        monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "detector_only")
        monkeypatch.setenv("FAIT_OBJECT_VERIFIER", "yolo")

        cfg = ScreenConfig(strategy="auto", verifier="auto", prompts=["knife"], gallery_dir="/tmp")
        s, v = _resolve_strategy_verifier(cfg)

        assert s == "detector_only"
        assert v == "yolo"

    def test_explicit_config_wins_over_env_when_not_auto(self, monkeypatch):
        monkeypatch.setenv("FAIT_OBJECT_STRATEGY", "detector_only")

        cfg = ScreenConfig(strategy="two_stage", verifier="yolo", prompts=["knife"], gallery_dir="/tmp")
        s, v = _resolve_strategy_verifier(cfg)

        # Explicit config should win when not "auto"
        assert s == "two_stage"


class TestScreenConfig:
    """Test ScreenConfig defaults and validation"""

    def test_default_thresholds(self):
        cfg = ScreenConfig(prompts=["knife"], gallery_dir="/tmp")
        assert cfg.fusion.iou_gate == 0.50
        assert cfg.fusion.rule in {"and", "weighted"}

    def test_custom_fusion_config(self):
        fusion = FusionConfig(iou_gate=0.7, rule="weighted", alpha=0.6)
        cfg = ScreenConfig(prompts=["knife"], gallery_dir="/tmp", fusion=fusion)
        assert cfg.fusion.iou_gate == 0.7
        assert cfg.fusion.alpha == 0.6

    def test_class_thresholds_default(self):
        cfg = ScreenConfig(prompts=["knife"], gallery_dir="/tmp")
        # Should have some default weapon thresholds
        assert "knife" in cfg.fusion.class_thresholds or "gun" in cfg.fusion.class_thresholds