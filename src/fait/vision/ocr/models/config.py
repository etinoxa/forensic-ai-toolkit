# src/fait/vision/ocr/config.py
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass, MISSING
from typing import Dict, List, Optional, Literal, Any, get_origin, get_args
import os, yaml, pathlib

# ----- Engine configs -----
@dataclass
class TesseractCfg:
    enabled: bool = True
    lang: str = "eng"
    tesseract_cmd: Optional[str] = None
    psm: Optional[int] = None
    oem: Optional[int] = None

@dataclass
class PaddleCfg:
    enabled: bool = True
    lang: str = "en"

@dataclass
class TrOCRCfg:
    enabled: bool = True
    model_id: str = "microsoft/trocr-base-printed"

@dataclass
class DonutCfg:
    enabled: bool = True
    model_id: str = "naver-clova-ix/donut-base"

@dataclass
class DocTRCfg:
    enabled: bool = True
    det_arch: str = "db_resnet50"
    reco_arch: str = "crnn_vgg16_bn"

# Bundle for service-level engine settings (optional convenience)
@dataclass
class OcrServiceCfg:
    tesseract: TesseractCfg = TesseractCfg()
    paddle:    PaddleCfg    = PaddleCfg()
    trocr:     TrOCRCfg     = TrOCRCfg()
    donut:     DonutCfg     = DonutCfg()
    doctr:     DocTRCfg     = DocTRCfg()

# ----- Fusion / strategy -----
Strategy = Literal["first_nonempty", "best_of", "consensus", "two_stage", "auto"]
Verifier = Literal["trocr", "donut", "tesseract", "paddleocr", "doctr", "none", "auto"]

@dataclass
class EngineCfg:
    # Generic knobs all models may use (extra keys in YAML will be ignored by the pipeline)
    @dataclass
    class EngineCfg:
        enabled: bool = True
        lang: Optional[str] = None
        model_id: Optional[str] = None
        det_arch: Optional[str] = None
        reco_arch: Optional[str] = None

        # Tesseract specifics
        tesseract_cmd: Optional[str] = None
        psm: Optional[int] = None
        oem: Optional[int] = None

        # PaddleOCR detection parameters - ADD THESE!
        det_db_thresh: Optional[float] = None
        det_db_box_thresh: Optional[float] = None
        det_db_unclip_ratio: Optional[float] = None

@dataclass
class FusionCfg:
    strategy: Strategy = "auto"
    verifier: Verifier = "auto"
    alpha: float = 0.6
    sim_tau: float = 0.70
    prefer_confidence: bool = True
    service: OcrServiceCfg = OcrServiceCfg()

# ----- Top-level OCR config used by pipeline -----
@dataclass
class OcrConfig:
    # IO
    gallery_dir: Optional[str] = None
    output_dir: Optional[str] = None
    found_csv: str = "found.csv"
    failures_csv: str = "failures.csv"

    # Pre-filters
    min_file_kb: int = 10
    min_dim_px: int = 100
    rotations: List[int] = field(default_factory=lambda: [0, 90, 180, 270])

    # Engines & order
    engines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    engine_order: List[str] = field(default_factory=lambda: ["trocr", "donut", "tesseract", "paddleocr", "doctr"])

    # Fusion
    fusion: FusionCfg = FusionCfg()

# ----- Dataclass rehydration helper -----
def _dc_from_dict(cls, data):
    """Recursively hydrate nested dataclasses from dicts."""
    if isinstance(data, cls):
        return data
    if not is_dataclass(cls) or not isinstance(data, dict):
        return data
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        val = data[f.name]
        ftype = f.type
        # Nested dataclass
        if is_dataclass(ftype):
            kwargs[f.name] = _dc_from_dict(ftype, val)
        # Dict/Mapping types - keep as-is
        elif get_origin(ftype) in (dict, Dict):
            kwargs[f.name] = val if isinstance(val, dict) else {}
        # List
        elif get_origin(ftype) in (list, List):
            (inner,) = get_args(ftype) if get_args(ftype) else (Any,)
            if isinstance(val, list) and is_dataclass(inner):
                kwargs[f.name] = [_dc_from_dict(inner, v) for v in val]
            else:
                kwargs[f.name] = val
        else:
            kwargs[f.name] = val
    return cls(**kwargs)

def _to_engine_cfg(d: dict | None) -> EngineCfg:
    if not d:
        return EngineCfg()
    # Only pass keys that exist in the dataclass
    allowed = set(EngineCfg.__annotations__.keys())
    return EngineCfg(**{k: v for k, v in d.items() if k in allowed})

def _apply_env_overrides(cfg: OcrConfig) -> None:
    """
    ENV wins only when fusion.strategy == 'auto' AND fusion.verifier == 'auto'
    (as requested). Safe overrides for a few common vars.
    """
    if cfg.fusion.strategy == "auto" and cfg.fusion.verifier == "auto":
        s = os.getenv("FAIT_OCR_STRATEGY", "").strip().lower()
        v = os.getenv("FAIT_OCR_VERIFIER", "").strip().lower()
        if s in {"first_nonempty", "best_of", "consensus", "two_stage"}:
            cfg.fusion.strategy = s
        if v in {"trocr", "donut", "tesseract", "paddleocr", "doctr", "none"}:
            cfg.fusion.verifier = v

    # Bonus: convenient Tesseract envs if provided
    tess_cmd = os.getenv("FAIT_TESSERACT_CMD")
    if tess_cmd and "tesseract" in cfg.engines:
        cfg.engines["tesseract"].tesseract_cmd = tess_cmd
    tess_lang = os.getenv("FAIT_TESSERACT_LANG")
    if tess_lang and "tesseract" in cfg.engines:
        cfg.engines["tesseract"].lang = tess_lang

def _to_fusion_cfg(d: Optional[dict]) -> FusionCfg:
    d = d or {}
    allowed = {f.name for f in fields(FusionCfg)}
    return FusionCfg(**{k: v for k, v in d.items() if k in allowed})

# ----- Loader -----
def load_ocr_config(yaml_path: str | pathlib.Path) -> OcrConfig:
    with open(yaml_path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    data = y.get("ocr", y) or {}
    engines_dict = data.get("engines") or {}
    engines = {name: _to_engine_cfg(cfg) for name, cfg in engines_dict.items()}
    fusion = _to_fusion_cfg(data.get("fusion"))

    return OcrConfig(
        gallery_dir=data.get("gallery_dir"),
        output_dir=data.get("output_dir"),
        min_file_kb=int(data.get("min_file_kb", 10)),
        min_dim_px=int(data.get("min_dim_px", 100)),
        rotations=[int(x) for x in (data.get("rotations") or [0, 90, 180, 270])],
        engines=engines,
        engine_order=list(data.get("engine_order") or ["trocr","donut","tesseract","paddleocr","doctr"]),
        fusion=fusion,
    )
