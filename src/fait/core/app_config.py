# src/fait/core/app_config.py
"""
Centralized application configuration.
Separates deployment settings (from .env) from algorithm behavior (from YAML).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List
import yaml
import os


@dataclass
class AudioStrategyConfig:
    """Audio speaker recognition strategy configuration"""
    strategy: str = "two_stage"
    detector: str = "titanet"
    tertiary: str = "none"

    # Fusion parameters
    fusion_method: str = "weighted"
    alpha: float = 0.60
    tau_star: float = 0.70

    # Model identifiers
    speechbrain_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    titanet_id: str = "nvidia/speakerverification_en_titanet_large"
    wavlm_id: str = "microsoft/wavlm-base-plus"


@dataclass
class VisionObjectConfig:
    """Object detection strategy configuration"""
    strategy: str = "detector_only"
    verifier: str = "deformable_detr"

    # Model identifiers
    gdino_model_id: str = "IDEA-Research/grounding-dino-base"
    yolo_model_id: str = "yolov8n.pt"
    detr_model_id: str = "SenseTime/deformable-detr"

    # GDINO thresholds
    gdino_box_threshold: float = 0.35
    gdino_text_threshold: float = 0.35
    gdino_nms_iou: float = 0.50
    gdino_long_side: int = 1024
    gdino_box_expand: float = 0.20

    # YOLO thresholds
    yolo_score_threshold: float = 0.25
    yolo_nms_iou: float = 0.50
    yolo_imgsz: int = 640

    # DETR thresholds
    detr_score_threshold: float = 0.25
    detr_nms_iou: float = 0.50

    # Fusion parameters
    fusion_rule: str = "and"
    fusion_alpha: float = 0.7
    fusion_tau_star: float = 0.60
    fusion_iou_gate: float = 0.50
    fusion_borderline_window: float = 0.05

    # Detector-only thresholds
    detector_only_default_tau: float = 0.50
    gdino_only_default_tau: float = 0.50


@dataclass
class VisionOCRConfig:
    """OCR strategy configuration"""
    strategy: str = "first_nonempty"
    verifier: str = "none"
    engine_order: List[str] = field(
        default_factory=lambda: ["paddle", "trocr", "donut", "tesseract", "doctr"])  # ← This must be here

    # Processing parameters
    min_file_kb: int = 10
    min_dim_px: int = 100
    rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])

    # Model identifiers
    trocr_model_id: str = "microsoft/trocr-large-printed"
    donut_model_id: str = "naver-clova-ix/donut-base"
    olmocr_model_id: str = "allenai/olmOCR-7B-0725"
    tesseract_lang: str = "eng"
    tesseract_cmd: Optional[str] = None
    paddle_lang: str = "en"
    paddle_det_db_thresh: float = 0.2
    paddle_det_db_box_thresh: float = 0.3
    paddle_det_db_unclip_ratio: float = 2.0
    doctr_det_arch: str = "db_resnet50"
    doctr_reco_arch: str = "crnn_vgg16_bn"


@dataclass
class VisionFaceConfig:
    """Face recognition configuration"""
    recognizer: str = "arcface"
    metric: str = "auto"
    thresholds: List[float] = field(default_factory=lambda: [0.80, 0.90])
    plot_results: bool = True


@dataclass
class VisionConfig:
    """Vision module configuration"""
    object_detection: VisionObjectConfig = field(default_factory=VisionObjectConfig)
    ocr: VisionOCRConfig = field(default_factory=VisionOCRConfig)
    face_recognition: VisionFaceConfig = field(default_factory=VisionFaceConfig)


@dataclass
class AudioConfig:
    """Audio module configuration"""
    speaker_recognition: AudioStrategyConfig = field(default_factory=AudioStrategyConfig)


@dataclass
class FaitConfig:
    """
    Main FAIT configuration.

    This contains application behavior (algorithms, models, strategies).
    Deployment settings (paths, logging, etc.) come from environment variables.
    """
    audio: AudioConfig = field(default_factory=AudioConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    @classmethod
    def load(cls, config_path: Optional[str | Path] = None) -> "FaitConfig":
        """
        Load configuration from YAML file.

        Environment variables can override specific settings, but YAML is the source of truth.
        This prevents .env pollution from affecting application behavior.

        Args:
            config_path: Path to YAML config. If None, uses default config.yaml in project root.

        Returns:
            Loaded configuration
        """
        if config_path is None:
            # Default to config.yaml in project root
            config_path = Path(__file__).parents[3] / "config.yaml"

        config_path = Path(config_path)

        if not config_path.exists():
            # Return defaults if no config file
            return cls()

        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # Parse nested structure
        config = cls(
            audio=AudioConfig(
                speaker_recognition=cls._parse_audio_speaker(data.get("audio", {}).get("speaker_recognition", {}))
            ),
            vision=VisionConfig(
                object_detection=cls._parse_vision_object(data.get("vision", {}).get("object_detection", {})),
                ocr=cls._parse_vision_ocr(data.get("vision", {}).get("ocr", {})),
                face_recognition=cls._parse_vision_face(data.get("vision", {}).get("face_recognition", {}))
            )
        )

        return config

    @staticmethod
    def _parse_audio_speaker(data: Dict[str, Any]) -> AudioStrategyConfig:
        """Parse audio speaker recognition config from dict"""
        return AudioStrategyConfig(
            strategy=data.get("strategy", "two_stage"),
            detector=data.get("detector", "titanet"),
            tertiary=data.get("tertiary", "none"),
            fusion_method=data.get("fusion", {}).get("method", "weighted"),
            alpha=float(data.get("fusion", {}).get("alpha", 0.60)),
            tau_star=float(data.get("fusion", {}).get("tau_star", 0.70)),
            speechbrain_id=data.get("models", {}).get("speechbrain_id", "speechbrain/spkrec-ecapa-voxceleb"),
            titanet_id=data.get("models", {}).get("titanet_id", "nvidia/speakerverification_en_titanet_large"),
            wavlm_id=data.get("models", {}).get("wavlm_id", "microsoft/wavlm-base-plus"),
        )

    @staticmethod
    def _parse_vision_object(data: Dict[str, Any]) -> VisionObjectConfig:
        """Parse vision object detection config from dict"""
        models = data.get("models", {})
        gdino = data.get("gdino", {})
        yolo = data.get("yolo", {})
        detr = data.get("detr", {})
        fusion = data.get("fusion", {})
        detector_only = data.get("detector_only", {})

        return VisionObjectConfig(
            strategy=data.get("strategy", "detector_only"),
            verifier=data.get("verifier", "deformable_detr"),
            gdino_model_id=models.get("gdino_model_id") or gdino.get("model_id", "IDEA-Research/grounding-dino-base"),
            yolo_model_id=models.get("yolo_model_id") or yolo.get("model_id", "yolov8n.pt"),
            detr_model_id=models.get("detr_model_id") or detr.get("model_id", "SenseTime/deformable-detr"),
            gdino_box_threshold=float(gdino.get("box_threshold", 0.35)),
            gdino_text_threshold=float(gdino.get("text_threshold", 0.35)),
            gdino_nms_iou=float(gdino.get("nms_iou", 0.50)),
            gdino_long_side=int(gdino.get("long_side", 1024)),
            gdino_box_expand=float(gdino.get("box_expand", 0.20)),
            yolo_score_threshold=float(yolo.get("score_threshold", 0.25)),
            yolo_nms_iou=float(yolo.get("nms_iou", 0.50)),
            yolo_imgsz=int(yolo.get("imgsz", 640)),
            detr_score_threshold=float(detr.get("score_threshold", 0.25)),
            detr_nms_iou=float(detr.get("nms_iou", 0.50)),
            fusion_rule=fusion.get("rule", "and"),
            fusion_alpha=float(fusion.get("alpha", 0.7)),
            fusion_tau_star=float(fusion.get("tau_star", 0.60)),
            fusion_iou_gate=float(fusion.get("iou_gate", 0.50)),
            fusion_borderline_window=float(fusion.get("borderline_window", 0.05)),
            detector_only_default_tau=float(detector_only.get("default_tau", 0.50)),
            gdino_only_default_tau=float(fusion.get("gdino_only_default_tau", 0.50)),
        )

    @staticmethod
    def _parse_vision_ocr(data: Dict[str, Any]) -> VisionOCRConfig:
        """Parse vision OCR config from dict"""
        models = data.get("models", {})
        engines_data = data.get("engines", {})

        return VisionOCRConfig(
            strategy=data.get("strategy", "first_nonempty"),
            verifier=data.get("verifier", "none"),
            engine_order=data.get("engine_order", ["paddle", "trocr", "donut", "tesseract"]),  # ← Use engine_order
            min_file_kb=int(data.get("min_file_kb", 10)),
            min_dim_px=int(data.get("min_dim_px", 100)),
            rotations=[int(x) for x in data.get("rotations", [0, 90, 180, 270])],
            trocr_model_id=models.get("trocr_model_id") or engines_data.get("trocr", {}).get("model_id",
                                                                                             "microsoft/trocr-large-printed"),
            donut_model_id=models.get("donut_model_id") or engines_data.get("donut", {}).get("model_id",
                                                                                             "naver-clova-ix/donut-base"),
            olmocr_model_id=models.get("olmocr_model_id") or engines_data.get("olmocr", {}).get("model_id",
                                                                                                "allenai/olmOCR-7B-0725"),

            tesseract_lang=models.get("tesseract_lang") or engines_data.get("tesseract", {}).get("lang", "eng"),
            tesseract_cmd=engines_data.get("tesseract", {}).get("tesseract_cmd"),
            paddle_lang=models.get("paddle_lang") or engines_data.get("paddle", {}).get("lang", "en"),
            paddle_det_db_thresh=float(engines_data.get("paddle", {}).get("det_db_thresh", 0.2)),
            paddle_det_db_box_thresh=float(engines_data.get("paddle", {}).get("det_db_box_thresh", 0.3)),
            paddle_det_db_unclip_ratio=float(engines_data.get("paddle", {}).get("det_db_unclip_ratio", 2.0)),
            doctr_det_arch=engines_data.get("doctr", {}).get("det_arch", "db_resnet50"),
            doctr_reco_arch=engines_data.get("doctr", {}).get("reco_arch", "crnn_vgg16_bn"),
        )

    @staticmethod
    def _parse_vision_face(data: Dict[str, Any]) -> VisionFaceConfig:
        """Parse vision face recognition config from dict"""
        return VisionFaceConfig(
            recognizer=data.get("recognizer", "arcface"),
            metric=data.get("metric", "auto"),
            thresholds=[float(t) for t in data.get("thresholds", [0.80, 0.90])],
            plot_results=bool(data.get("plot_results", True)),
        )


# Singleton for application-wide config
_app_config: Optional[FaitConfig] = None


def get_app_config(reload: bool = False) -> FaitConfig:
    """
    Get the application configuration singleton.

    Args:
        reload: If True, reload from disk

    Returns:
        Application configuration
    """
    global _app_config
    if _app_config is None or reload:
        _app_config = FaitConfig.load()
    return _app_config