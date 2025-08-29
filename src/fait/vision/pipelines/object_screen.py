# src/fait/vision/pipelines/object_screen.py  (only the changed bits)

from typing import List, Dict, Optional, Tuple, Literal
import os, json, time, shutil, re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
import logging
import torch

from PIL import Image

from fait.core.paths import get_paths
from fait.core.utils import ProgressMeter, ensure_folder, file_md5, write_object_report
from fait.vision.services.object_service import get_object_service
from fait.vision.detectors.grounding_dino import GDINOConfig
from fait.vision.detectors.deformable_detr import DefDETRConfig
from fait.vision.detectors.yolo import YoloConfig

log = logging.getLogger("fait.vision.pipelines.object_screen")

def _append_jsonl(log_path, obj) -> None:
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

def _sanitize(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[^\w.\-]+", "_", str(s))
    return s[:maxlen].strip("_")

def _object_run_dir_name(strategy, cfg, device: str, gd=None, cd=None) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dev = (device or ("cuda" if torch.cuda.is_available() else "cpu")).lower()

    if strategy == "gdino_only":
        tag = f"GDINO__{_sanitize(cfg.gdino.model_id)}__{dev}__{ts}"
    elif strategy == "detector_only":
        det_tag = "YOLO" if cfg.verifier == "yolo" else "DETR"
        mid = cfg.yolo.model_id if cfg.verifier == "yolo" else cfg.detr.model_id
        tag = f"{det_tag}__{_sanitize(mid)}__{dev}__{ts}"
    else:  # two_stage
        det_tag = "YOLO" if cfg.verifier == "yolo" else "DETR"
        tag = f"GDINO+{det_tag}__{_sanitize(cfg.gdino.model_id)}__{_sanitize((cfg.yolo.model_id if cfg.verifier=='yolo' else cfg.detr.model_id))}__{dev}__{ts}"
    return tag

@dataclass
class FusionConfig:
    iou_gate: float = 0.50
    rule: str = "and"          # "and" or "weighted"
    alpha: float = 0.5
    tau_star: float = 0.6
    class_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "gun": 0.40, "handgun": 0.40, "pistol": 0.40, "revolver": 0.40,
        "knife": 0.45, "kitchen knife": 0.45, "chef knife": 0.45
    })
    borderline_window: float = 0.05
    gdino_only_default_tau: float = 0.5  # used when verifier == "none"

@dataclass
class DetectorOnlyConfig:
    """Thresholds for detector-only mode."""
    default_tau: float = 0.50                     # accept if score >= default_tau
    class_thresholds: Dict[str, float] = field(default_factory=dict)  # optional per-class

@dataclass
class ScreenConfig:
    prompts: List[str]
    gallery_dir: str
    output_dir: Optional[str] = None
    save_crops: bool = False
    # progress UI: "tqdm", "log", or "none"
    progress: Literal["log", "none"] = "log"
    progress_every: int = 10  # used when progress == "log"

    # now supports three modes
    strategy: Literal["gdino_only", "detector_only", "two_stage", "auto"] = "two_stage"
    verifier: Literal["none", "deformable_detr", "yolo", "auto"] = "deformable_detr"

    gdino: GDINOConfig = GDINOConfig()
    detr: DefDETRConfig = DefDETRConfig()
    yolo: YoloConfig = YoloConfig()
    fusion: FusionConfig = FusionConfig()
    detector_only: DetectorOnlyConfig = DetectorOnlyConfig()

    # optional custom run name
    run_name: Optional[str] = None

def _resolve_strategy_verifier(cfg: ScreenConfig) -> tuple[str, str]:
    s = (cfg.strategy or "auto").strip().lower()
    v = (cfg.verifier or "auto").strip().lower()
    env_s = os.getenv("FAIT_OBJECT_STRATEGY", "").strip().lower()
    env_v = os.getenv("FAIT_OBJECT_VERIFIER", "").strip().lower()

    # 1) Strategy: env overrides when cfg is "auto"
    if s == "auto":
        s = env_s or "two_stage"  # sensible default if nothing set

    # 2) Verifier: env overrides when cfg is "auto"
    if v == "auto":
        v = env_v or ("none" if s == "gdino_only" else "yolo")

    # 3) Clamp combos
    if s == "gdino_only":
        v = "none"
    if s in {"detector_only", "two_stage"} and v not in {"yolo", "deformable_detr"}:
        v = "yolo"  # fallback

    return s, v

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _iou_xyxy(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union

def _pick_class_threshold(fcfg: FusionConfig, label: str) -> float:
    return fcfg.class_thresholds.get(label.lower(), fcfg.gdino_only_default_tau)

def _crop(img: Image.Image, xyxy) -> Image.Image:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(img.width, x2); y2 = min(img.height, y2)
    return img.crop((x1, y1, x2, y2))

def run_object_screen(cfg: ScreenConfig) -> Dict:
    t0 = time.time()
    paths = get_paths()

    env_prog = os.getenv("FAIT_PROGRESS", "").strip().lower()
    if env_prog in {"tqdm", "log", "none"}:
        cfg.progress = env_prog

    strategy, cfg.verifier = _resolve_strategy_verifier(cfg)
    log.info("object_screen:resolved", extra={"strategy": strategy, "verifier": cfg.verifier})

    _made_found = _made_review = _made_crops = False

    # init detectors
    svc = get_object_service()
    # --- clamp verifier for pure GDINO runs (prevents two-stage path) ---
    if strategy == "gdino_only":
        cfg.verifier = "none"

    # --- build via pooled service (one init per model per process) ---
    gd = svc.get_gdino(cfg.gdino) if strategy in {"gdino_only", "two_stage"} else None

    cd = None
    if strategy in {"detector_only", "two_stage"}:
        if cfg.verifier == "yolo":
            cd = svc.get_yolo(cfg.yolo)
        elif cfg.verifier == "deformable_detr":
            cd = svc.get_detr(cfg.detr)
        else:
            raise ValueError("detector_only/two_stage requires verifier 'yolo' or 'deformable_detr'")

    if strategy == "detector_only" and cd is None:
        raise RuntimeError("detector_only strategy requires cfg.verifier in {'yolo','deformable_detr'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = Path(cfg.output_dir) if cfg.output_dir else (paths.outputs / "object")
    ensure_folder(base)
    run_name = cfg.run_name.strip() if cfg.run_name else _object_run_dir_name(strategy, cfg, device)
    run_dir = base / run_name

    log_path = run_dir / "log.jsonl"
    found_dir = run_dir / "found_images"
    review_dir = run_dir / "review_queue"
    crops_dir  = run_dir / "crops"

    wrote_anything = False
    log_file = None

    try:
        # Create run_dir and open log only after models are ready
        ensure_folder(run_dir)

        # ... main loop ...
        # when you actually copy a found image or write a crop:
        #   ensure_folder(found_dir); shutil.copy(...); wrote_anything = True
        #   ensure_folder(crops_dir);  ...; wrote_anything = True
        # if you append to JSONL or write report: wrote_anything = True

        # at the end, write report.txt -> wrote_anything = True

    except Exception as e:
        # Clean up if nothing meaningful was created
        try:
            if run_dir.exists() and not wrote_anything:
                shutil.rmtree(run_dir, ignore_errors=True)
        except Exception:
            pass

        # re-raise so caller sees the error
        raise






    log.info("object_screen:models_ready", extra={
        "strategy": strategy,
        "verifier": cfg.verifier,
        "gdino_built": bool(gd),
        "detector": (type(cd).__name__ if cd else None),
    })

    log.info("object_screen:start", extra={
        "strategy": strategy,
        "verifier": cfg.verifier,
        "gdino_built": bool(gd),
        "prompts": cfg.prompts,
        "detector": (type(cd).__name__ if cd else None),
    })

    log.info("object_screen:config", extra={
        "verifier": cfg.verifier,
        "env_verifier": os.getenv("FAIT_OBJECT_VERIFIER"),
        "gallery_dir": cfg.gallery_dir,
        "prompts": cfg.prompts,
        "gdino": {
            "model_id": cfg.gdino.model_id,
            "box_threshold": cfg.gdino.box_threshold,
            "text_threshold": cfg.gdino.text_threshold,
            "long_side": cfg.gdino.long_side,
            "box_expand": cfg.gdino.box_expand,
            "nms_iou": cfg.gdino.nms_iou,
            "normalize_prompts": cfg.gdino.normalize_prompts,
            "add_relational_prompts": cfg.gdino.add_relational_prompts,
        },
        "fusion": {
            "rule": cfg.fusion.rule,
            "iou_gate": cfg.fusion.iou_gate,
            "alpha": cfg.fusion.alpha,
            "tau_star": cfg.fusion.tau_star,
            "gdino_only_default_tau": cfg.fusion.gdino_only_default_tau,
            "borderline_window": cfg.fusion.borderline_window,
            "class_thresholds": cfg.fusion.class_thresholds,
        },
    })

    try:
        log.debug("object_screen:cfg_dump", extra={
            "ScreenConfig": {
                "gdino": asdict(cfg.gdino),
                "fusion": asdict(cfg.fusion),
                "verifier": cfg.verifier,
                "yolo": asdict(cfg.yolo),
                "detr": asdict(cfg.detr),
            }
        })
    except Exception:
        pass

    processed = found = review = 0
    gallery = sorted([p for p in Path(cfg.gallery_dir).iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}])

    total = len(gallery)
    pm = ProgressMeter(
        total=total,
        label="object_screen:progress",
        logger=log,
        log_path=log_path,  # your .../log.jsonl path; or None to skip JSONL
        emit_every_n=5,
        emit_every_sec=2.0,
    )

    for fpath in gallery:
        processed += 1
        img = Image.open(fpath).convert("RGB")
        pm.set_counts(processed, found=found, review=review)

        # ================== DETECTOR-ONLY ==================
        if strategy == "detector_only":
            dets = cd.detect(img) if cd else []
            if not dets:
                continue

            accepted = False
            borderline_hit = False
            best_entry = None

            for det in dets:
                label = str(det["label"]).lower()
                score = float(det["score"])
                tau = cfg.detector_only.class_thresholds.get(label, cfg.detector_only.default_tau)

                entry = {
                    "image": str(fpath),
                    "hash": file_md5(fpath),
                    "prompt": None,
                    "gdino_score": None,
                    "detector_label": label,
                    "detector_score": score,
                    "iou": None,
                    "fused": score,              # single-model score = fused
                    "threshold": tau,
                    "gdino_box": None,
                    "detector_mapped_box": det.get("box"),
                    "model_versions": {"verifier": type(cd).__name__},
                    "time": datetime.now().isoformat(timespec="seconds"),
                }

                if score >= tau:
                    accepted = True; best_entry = entry; break
                elif cfg.fusion.borderline_window and cfg.fusion.borderline_window > 0 \
                        and abs(score - tau) <= cfg.fusion.borderline_window:
                    borderline_hit = True; best_entry = best_entry or entry

            if accepted and best_entry:
                if not _made_found: ensure_folder(found_dir); _made_found = True
                shutil.copy2(fpath, found_dir / fpath.name)
                if cfg.save_crops:
                    if not _made_crops: ensure_folder(crops_dir); _made_crops = True
                    if best_entry.get("detector_mapped_box"):
                        _crop(img, best_entry["detector_mapped_box"]).save(crops_dir / f"{Path(fpath).stem}_crop.jpg")
                found += 1
            elif borderline_hit and best_entry and (cfg.fusion.borderline_window and cfg.fusion.borderline_window > 0):
                if not _made_review: ensure_folder(review_dir); _made_review = True
                shutil.copy2(fpath, review_dir / fpath.name)
                review += 1
            continue  # next image

        # proposals from GDINO (already normalized/enriched inside the detector)
        props = gd.propose(img, cfg.prompts)

        # optional: quick debug of top-3 scores per image
        if props:
            top_scores = sorted((p["score"] for p in props), reverse=True)[:3]
            log.debug("gdino:top_scores", extra={"image": str(fpath), "scores": [round(s, 3) for s in top_scores]})

        if not props:
            continue

        accepted = False
        borderline_hit = False
        best_entry = None

        for prop in props:
            g = float(prop["score"])
            lab = prop["prompt"].lower()
            # --- GDINO-ONLY MODE -------------------------------------------
            if cfg.verifier == "none":
                tau = _pick_class_threshold(cfg.fusion, lab)
                entry = {
                    "image": str(fpath),
                    "hash": file_md5(fpath),
                    "prompt": prop["prompt"],
                    "gdino_score": g,
                    "detector_label": None,
                    "detector_score": None,
                    "iou": 1.0,
                    "fused": g,
                    "threshold": tau,
                    "gdino_box": prop["box"],
                    "detector_mapped_box": None,
                    "model_versions": {"grounding_dino": gd.cfg.model_id, "verifier": "none"},
                    "time": datetime.now().isoformat(timespec="seconds"),
                }

                if g >= tau:
                    accepted = True; best_entry = entry; break
                elif cfg.fusion.borderline_window and cfg.fusion.borderline_window > 0 \
                         and abs(g - tau) <= cfg.fusion.borderline_window:
                    borderline_hit = True; best_entry = best_entry or entry
                continue

            # --- TWO-STAGE MODE (unchanged) --------------------------------
            roi = _crop(img, prop["box"])
            dets = cd.detect(roi)
            if not dets:
                continue

            px1, py1, px2, py2 = prop["box"]
            for det in dets:
                dx1, dy1, dx2, dy2 = det["box"]
                mapped = [px1 + dx1, py1 + dy1, px1 + dx2, py1 + dy2]
                iou = _iou_xyxy(prop["box"], mapped)
                if iou < cfg.fusion.iou_gate:
                    continue

                label = det["label"].lower()
                gscore = g; cscore = float(det["score"])

                if cfg.fusion.rule == "and":
                    tau = _pick_class_threshold(cfg.fusion, label)
                    ok = (gscore >= tau) and (cscore >= tau)
                    fused = min(gscore, cscore)
                else:
                    fused = cfg.fusion.alpha * gscore + (1 - cfg.fusion.alpha) * cscore
                    tau = cfg.fusion.tau_star
                    ok = fused >= tau

                entry = {
                    "image": str(fpath),
                    "hash": file_md5(fpath),
                    "prompt": prop["prompt"],
                    "gdino_score": gscore,
                    "detector_label": label,
                    "detector_score": cscore,
                    "iou": iou,
                    "fused": fused,
                    "threshold": tau,
                    "gdino_box": prop["box"],
                    "detector_mapped_box": mapped,
                    "model_versions": {
                        "grounding_dino": gd.cfg.model_id,
                        "verifier": type(cd).__name__,
                    },
                    "time": datetime.now().isoformat(timespec="seconds"),
                }


                if ok:
                    accepted = True; best_entry = entry; break
                elif cfg.fusion.borderline_window and cfg.fusion.borderline_window > 0 \
                         and abs(fused - tau) <= cfg.fusion.borderline_window:
                    borderline_hit = True; best_entry = best_entry or entry

            if accepted:
                break

        # outputs (lazy subfolders)
        if accepted and best_entry:
            if not _made_found: ensure_folder(found_dir); _made_found = True
            shutil.copy2(fpath, found_dir / fpath.name)
            if cfg.save_crops:
                if not _made_crops: ensure_folder(crops_dir); _made_crops = True
                crop_box = best_entry.get("detector_mapped_box") or best_entry.get("gdino_box")
                if crop_box:
                    _crop(img, crop_box).save(crops_dir / f"{Path(fpath).stem}_crop.jpg")
            found += 1
        elif borderline_hit and best_entry:
            if not _made_review: ensure_folder(review_dir); _made_review = True
            shutil.copy2(fpath, review_dir / fpath.name)
            review += 1

    # end of run
    pm.close()

    dt = time.time() - t0
    log.info("object_screen:done", extra={
        "processed": processed, "found": found, "review": review, "secs": round(dt, 2),
        "run_dir": str(run_dir)
    })

    out = {
        "processed": processed,
        "found": found,
        "review": review,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
    }

    models: Dict[str, str] = {}
    if strategy in {"gdino_only", "two_stage"}:
        models["gdino"] = cfg.gdino.model_id
    if strategy in {"detector_only", "two_stage"}:
        if cfg.verifier == "yolo":
            models["yolo"] = cfg.yolo.model_id
        elif cfg.verifier == "deformable_detr":
            models["detr"] = cfg.detr.model_id

    thresholds: Dict[str, float] = {}
    if strategy in {"gdino_only", "two_stage"}:
        thresholds["BOX_THRESHOLD"] = float(cfg.gdino.box_threshold)
        thresholds["TEXT_THRESHOLD"] = float(cfg.gdino.text_threshold)
        thresholds["GDINO_NMS_IOU"] = float(getattr(cfg.gdino, "nms_iou", 0.5))

    if strategy in {"detector_only", "two_stage"}:
        if cfg.verifier == "yolo":
            thresholds["YOLO_SCORE_TH"] = float(cfg.yolo.score_threshold)
            thresholds["YOLO_NMS_IOU"] = float(cfg.yolo.nms_iou)
        elif cfg.verifier == "deformable_detr":
            thresholds["DETR_SCORE_TH"] = float(cfg.detr.score_threshold)
            thresholds["DETR_NMS_IOU"] = float(cfg.detr.nms_iou)

    # acceptance (numeric only)
    if strategy == "gdino_only":
        thresholds["ACCEPT_TAU"] = float(cfg.fusion.gdino_only_default_tau)
    elif strategy == "detector_only":
        thresholds["ACCEPT_TAU"] = float(cfg.detector_only.default_tau)
    else:  # two_stage
        thresholds["ALPHA"] = float(cfg.fusion.alpha)
        thresholds["TAU_STAR"] = float(cfg.fusion.tau_star)
        thresholds["IOU_GATE"] = float(cfg.fusion.iou_gate)

    counts = {"processed": processed, "found": found, "review": review}
    found_files = sorted([p.name for p in (run_dir / "found_images").glob("*") if p.is_file()])

    # device string (same as used in run-dir naming)
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    report_path = write_object_report(
        output_dir=run_dir,
        strategy=strategy,
        device=device_str,
        prompts=[str(p) for p in cfg.prompts],
        models=models,
        thresholds=thresholds,
        counts=counts,
        found_files=found_files,
        filename="report.txt",
    )

    out["report_path"] = report_path
    return out

