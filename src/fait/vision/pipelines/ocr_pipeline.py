# src/fait/vision/pipelines/ocr_pipeline.py
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from fait.core.paths import get_paths
from fait.core.utils import (
    ensure_folder,
    is_image_file,
    human_size,
    ProgressMeter,
    resolve_strategy_verifier,
    resolve_engine_order,
    write_report_ocr,
)
# One true config types & loader
from fait.vision.ocr.config import OcrConfig, FusionCfg, load_ocr_config
# Engine factory (do NOT import the module named `engines` to avoid name clashes)
import fait.vision.ocr.engines as engreg

# if not hasattr(get_engine, "get_engine"):
#     from fait.vision.ocr.engines import get_engine as _get_engine
#     get_engine.get_engine = _get_engine



import logging
log = logging.getLogger("fait.vision.pipelines.ocr")


# ---------- strategy helpers ----------

# maps *_only strategies → normalized engine key
_ONLY_MAP = {
    "tesseract_only": "tesseract",
    "paddle_only":    "paddle",
    "trocr_only":     "trocr",
    "donut_only":     "donut",
    "doctr_only":     "doctr",
}

def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _avg_conf(conf) -> Optional[float]:
    if conf is None:
        return None
    if isinstance(conf, (int, float)):
        return float(conf)
    try:
        vals = [float(x) for x in conf if x is not None]
        return float(sum(vals) / max(1, len(vals))) if vals else None
    except Exception:
        return None


def _normalize_engine_result(res) -> Tuple[str, Optional[float]]:
    """
    Accepts dicts like {'text':..., 'confidence':...} or tuples (text, conf) or just text.
    Returns (text:str, confidence: Optional[float]).
    """
    if res is None:
        return "", None
    if isinstance(res, tuple) and len(res) == 2:
        text, conf = res
        return str(text or "").strip(), _avg_conf(conf)
    if isinstance(res, dict):
        text = str(res.get("text", "") or "").strip()
        conf = _avg_conf(res.get("confidence"))
        return text, conf
    return str(res or "").strip(), None


def _should_skip(path: Path, min_kb: int, min_dim: int) -> Tuple[bool, str]:
    try:
        size = path.stat().st_size
        if size < min_kb * 1024:
            return True, f"file <{min_kb}KB ({human_size(size)})"
    except Exception:
        return True, "stat_failed"

    try:
        with Image.open(path) as im:
            w, h = im.size
        # Accept an int or a pair [mw, mh]
        if isinstance(min_dim, (list, tuple)) and len(min_dim) >= 2:
            mw, mh = int(min_dim[0]), int(min_dim[1])
        else:
            mw = mh = int(min_dim)
        if w < mw or h < mh:
            return True, f"too_small ({w}x{h} < {mw}x{mh}px)"
    except Exception:
        return True, "open_failed"

    return False, ""


def _eng_to_dict(e) -> dict:
    """Normalize engine cfg objects to plain dicts for .get() access."""
    if isinstance(e, dict):
        return e
    return {
        "enabled": getattr(e, "enabled", True),
        "lang": getattr(e, "lang", "auto"),
        "model_id": getattr(e, "model_id", None),
        "tesseract_cmd": getattr(e, "tesseract_cmd", None),
        "psm": getattr(e, "psm", None),
        "oem": getattr(e, "oem", None),
        "det_arch": getattr(e, "det_arch", None),
        "reco_arch": getattr(e, "reco_arch", None),
    }


# ---------- public entry points ----------

def run_ocr_from_yaml(yaml_path: str | Path, gallery_dir: str | Path | None = None) -> Dict:
    cfg = load_ocr_config(str(yaml_path))
    if gallery_dir:
        cfg.gallery_dir = str(gallery_dir)
    return run_ocr(cfg)


def run_ocr(cfg: OcrConfig) -> Dict:
    """
    Runs OCR with rotations and fusion. Writes:
      - found.csv
      - failures.csv
      - log.jsonl
    Returns a summary dict with paths and counts.
    """
    t0 = time.time()
    paths = get_paths()

    # Point Paddle caches to project cache BEFORE any engine construction
    try:
        ocr_cache = paths.models_cache / "ocr"
        ensure_folder(ocr_cache)
        os.environ["PADDLEX_HOME"] = str(ocr_cache)
        os.environ["PPOCR_HOME"] = str(ocr_cache)
        os.environ.setdefault("PADDLEHUB_HOME", str(ocr_cache))
    except Exception:
        pass

    # Normalize nested config
    if isinstance(getattr(cfg, "fusion", None), dict):
        cfg.fusion = FusionCfg(**cfg.fusion)  # type: ignore[assignment]
    if isinstance(getattr(cfg, "engines", None), dict):
        cfg.engines = {name: _eng_to_dict(val) for name, val in cfg.engines.items()}  # type: ignore[assignment]

    # --- normalize engine-name aliases (so YAML can say 'paddleocr', 'donot', etc.) ---
    _alias = {"paddleocr": "paddle", "donot": "donut"}
    cfg.engines = {_alias.get(k.lower(), k.lower()): v for k, v in (cfg.engines or {}).items()}

    # If YAML specified engine_order with aliases, normalize that too
    if getattr(cfg, "engine_order", None):
        cfg.engine_order = [_alias.get(x.lower(), x.lower()) for x in cfg.engine_order]

    # Resolve dirs
    gallery_dir = Path(cfg.gallery_dir) if cfg.gallery_dir else Path(
        os.getenv("OCR_GALLERY_DIR", paths.repo_root / "datasets/images/text_ocr")
    )
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery not found: {gallery_dir}")

    # Strategy/verifier resolution (ENV wins only when YAML=auto)
    strategy, verifier = resolve_strategy_verifier(cfg.fusion)
    # If the cfg object has explicit .strategy / .verifier attributes set, honor them.
    if hasattr(cfg, "strategy") and isinstance(getattr(cfg, "strategy"), str) and cfg.strategy:
        strategy = cfg.strategy.strip().lower()
    if hasattr(cfg, "verifier") and isinstance(getattr(cfg, "verifier"), str) and cfg.verifier:
        verifier = cfg.verifier.strip().lower()

    # -------- engine order (enabled only) --------
    order = resolve_engine_order(cfg, strategy)
    if not order:
        raise RuntimeError("No OCR engines available/enabled after config filtering.")

    enabled = [k for k, v in (cfg.engines or {}).items() if v.get("enabled", True)]
    log.info("%s %s", "ocr:engines_enabled", json.dumps({
        "enabled": enabled,
        "strategy": strategy,
        "verifier": verifier,
        "order": order,
    }))

    run_dir, _ = write_report_ocr(
        paths.outputs,
        strategy=strategy,
        engines=order,
        verifier=verifier,
        counts={"processed": 0, "found": 0, "review": 0},
    )

    # Now set fixed file paths inside that folder
    found_csv = run_dir / "found.csv"
    failures_csv = run_dir / "failures.csv"
    jsonl_log = run_dir / "log.jsonl"

    # Start log (now we know run_dir)
    log.info("ocr:start", extra={
        "gallery": str(gallery_dir),
        "run_dir": str(run_dir),
        "strategy": strategy,
        "verifier": verifier,
        "min_file_kb": cfg.min_file_kb,
        "min_dim_px": cfg.min_dim_px,
        "rotations": cfg.rotations,
        "engine_order": cfg.engine_order,
    })

    # --------- LAZY engine getter: create an engine only when it is used ---------
    _engines_cache: Dict[str, object] = {}

    def _eng(name: str):
        if name not in _engines_cache:
            _engines_cache[name] = engreg.get_engine(name, cfg.engines.get(name))
        return _engines_cache[name]

    # ------------- strategies -------------
    allowed_verifiers_ts = {"tesseract", "trocr", "doctr", "donut"}  # two_stage
    allowed_verifiers_det = {"paddle", "tesseract", "trocr", "doctr", "donut"}  # detector_only

    def _run_two_stage(img):
        # primary: Paddle (required)
        primary = "paddle"
        if primary not in cfg.engines or not cfg.engines.get(primary, {}).get("enabled", True):
            raise RuntimeError("two_stage requires Paddle (primary) enabled.")

        # read verifier choice (validated elsewhere)
        if verifier not in {"tesseract", "trocr", "doctr", "donut"}:
            raise RuntimeError("two_stage verifier must be one of {'tesseract','trocr','doctr','donut'}")

        # languages: prefer engine-specific lang, default to "auto"
        lang_paddle = (cfg.engines.get("paddle") or {}).get("lang", "auto")
        lang_verifier = (cfg.engines.get(verifier) or {}).get("lang", "auto")

        # 1) Paddle first
        p_text, p_conf = _normalize_engine_result(_eng(primary).ocr(img, lang=lang_paddle))

        # 2) Verifier tries to beat it
        v_text, v_conf = _normalize_engine_result(_eng(verifier).ocr(img, lang=lang_verifier))

        # policy: prefer verifier when non-empty; else fall back to paddle
        if v_text:
            return v_text, v_conf, lang_verifier, f"two_stage:{primary}→{verifier}"
        return p_text, p_conf, lang_paddle, f"two_stage:{primary}"

    def _direct_paddle_detect_and_ocr(img, verifier_name, lang):
        """Direct PaddleOCR call bypassing the engine wrapper"""
        from paddleocr import PaddleOCR
        import numpy as np

        # Initialize PaddleOCR directly with working parameters
        paddle = PaddleOCR(
            use_angle_cls=False,
            lang='en',
            det_db_thresh=0.2,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0
        )

        # Run OCR
        result = paddle.ocr(np.array(img))

        # Parse results
        if result and isinstance(result, list) and result[0]:
            if isinstance(result[0], dict):
                # New format
                texts = result[0].get('rec_texts', [])
                scores = result[0].get('rec_scores', [])
                if texts:
                    text = ' '.join(texts)
                    conf = sum(scores) / len(scores) if scores else None
                    return text, conf, lang, f"direct_paddle"
            else:
                # Old format
                texts = []
                for item in result[0]:
                    if len(item) >= 2 and item[1] and len(item[1]) >= 1:
                        texts.append(str(item[1][0]))
                if texts:
                    return ' '.join(texts), None, lang, f"direct_paddle"

        return "", None, lang, "direct_paddle:no_text"

    def _run_detector_only(img, lang):
        """Run detection with Paddle, recognition with verifier"""
        if verifier not in allowed_verifiers_det:
            raise RuntimeError(
                f"detector_only verifier must be one of {sorted(allowed_verifiers_det)}; got {verifier!r}")
        if "paddle" not in cfg.engines or not cfg.engines.get("paddle", {}).get("enabled", True):
            raise RuntimeError("detector_only requires Paddle (detector) enabled.")

        # 1) detect
        det = _eng("paddle")
        boxes = det.detect(img)
        log.info("paddle:detect", extra={"boxes": len(boxes) if boxes else 0})

        if not boxes:
            # Try fallback: full-page OCR with verifier
            log.info("detector_only:fallback_fullpage")
            v_text, v_conf = _normalize_engine_result(_eng(verifier).ocr(img, lang=lang))
            if v_text:
                return v_text, v_conf, lang, "detector_only:fallback_fullpage"
            return "", None, lang, "detector_only:no_text"

        # 2) recognize each crop
        recog = _eng(verifier)
        parts, confs = [], []

        log.info(f"Processing {len(boxes)} detected regions with {verifier}")

        for i, (_, crop) in enumerate(boxes):
            try:
                # Log crop info
                if i < 3:  # Log first few crops
                    log.info(f"Crop {i}: size={crop.size}")

                # Try recognize method first if available
                if hasattr(recog, 'recognize') and callable(recog.recognize):
                    out = recog.recognize(crop)
                    if out is not None and hasattr(out, 'text'):
                        if out.text.strip():
                            parts.append(out.text.strip())
                            if hasattr(out, 'confidence') and out.confidence is not None:
                                confs.append(float(out.confidence))
                            log.info(f"Crop {i} recognized: {out.text[:30]}...")
                        continue

                # Fallback to ocr method
                result = recog.ocr(crop, lang=lang)
                t, c = _normalize_engine_result(result)
                if t:
                    parts.append(t.strip())
                    if c is not None:
                        confs.append(float(c))
                    log.info(f"Crop {i} OCR'd: {t[:30]}...")
            except Exception as e:
                log.error(f"Error recognizing crop {i}: {e}")

        log.info(f"Recognition complete: {len(parts)} text regions found from {len(boxes)} boxes")

        text = " ".join(parts).strip()
        avg = (sum(confs) / len(confs)) if confs else None

        return text, avg, lang, f"detector_only:paddle→{verifier}"


    found_f = found_csv.open("w", newline="", encoding="utf-8")
    fail_f = failures_csv.open("w", newline="", encoding="utf-8")
    log_f = jsonl_log.open("w", encoding="utf-8")
    found_w = csv.writer(found_f)
    fail_w = csv.writer(fail_f)
    found_w.writerow(["file", "text", "lang", "confidence", "notes"])
    fail_w.writerow(["file", "reason"])

    # Files to process
    files = [p for p in sorted(gallery_dir.iterdir()) if p.is_file() and is_image_file(p)]
    total = len(files)
    processed = 0
    found = 0
    failed = 0

    pm = ProgressMeter(
        total=total,
        label="ocr:progress",
        logger=log,
        log_path=jsonl_log,
        emit_every_n=10,
        emit_every_sec=2.0,
    )


    def _jsonl(entry: dict) -> None:
        log_f.write(json.dumps(entry) + "\n"); log_f.flush()

    try:
        for fp in files:
            processed += 1

            # Skip rules
            skip, reason = _should_skip(fp, cfg.min_file_kb, cfg.min_dim_px)
            if skip:
                fail_w.writerow([fp.name, reason])
                failed += 1
                pm.set_counts(processed, found=found, review=0)
                _jsonl({"file": fp.name, "action": "skipped", "reason": reason})
                continue


            # Try rotations in order; stop at first non-empty text given fusion policy
            best_text, best_conf, best_notes, best_lang = "", None, "", None

            with Image.open(fp) as base_img:
                for rot in (cfg.rotations or [0]):
                    try:
                        img = base_img.rotate(rot, expand=True) if rot else base_img
                    except Exception:
                        continue

                    results: List[Tuple[str, Optional[float], str, str]] = []  # (text, conf, engine, lang_used)

                    if strategy == "first_nonempty":
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(_eng(name).ocr(img, lang=lang))
                            if text:
                                best_text, best_conf, best_notes, best_lang = text, conf, f"{name};rot={rot}", lang
                                break

                    elif strategy == "best_of":
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(_eng(name).ocr(img, lang=lang))
                            if text:
                                results.append((text, conf, name, lang))
                        if results:
                            # pick max confidence; if all conf None, pick the first
                            pick = max(results, key=lambda r: (r[1] is not None, r[1] or 0.0))
                            best_text, best_conf, best_notes, best_lang = pick[0], pick[1], f"{pick[2]};rot={rot}", pick[3]

                    elif strategy == "consensus":
                        for name in order:
                            lang = cfg.engines[name].get("lang", "auto")
                            text, conf = _normalize_engine_result(_eng(name).ocr(img, lang=lang))
                            if text:
                                results.append((text, conf, name, lang))
                        if results:
                            # simple consensus: prefer highest confidence; if confs missing, prefer first
                            results_sorted = sorted(results, key=lambda r: (r[1] is not None, r[1] or 0.0), reverse=True)
                            pick = results_sorted[0]
                            best_text, best_conf, best_notes, best_lang = pick[0], pick[1], f"{pick[2]};rot={rot}", pick[3]
                            # fallback: if low conf and there exists another near-duplicate text, keep highest conf

                    elif strategy == "two_stage":
                        text, conf, lang_used, notes = _run_two_stage(img)
                        log.info("ocr:stage_out", extra={"strategy": strategy, "rot": rot, "len": len(text or "")})
                        if text:
                            best_text, best_conf, best_lang, best_notes = text, conf, lang_used, f"{notes};rot={rot}"
                            break  # two-stage is single-result per rotation

                    #Works with _direct_paddle_detect_and_ocr
                    # elif strategy == "detector_only":
                    #     lang = (cfg.engines.get(verifier) or {}).get("lang", "auto")
                    #
                    #     # Use direct PaddleOCR instead of engine wrapper
                    #     text, conf, lang_used, notes = _direct_paddle_detect_and_ocr(img, verifier, lang)
                    #
                    #     log.info("ocr:stage_out", extra={"strategy": strategy, "rot": rot, "len": len(text or "")})
                    #     if text:
                    #         best_text, best_conf, best_lang, best_notes = text, conf, lang_used, f"{notes};rot={rot}"

                    elif strategy == "detector_only":
                        # Add this line to define lang
                        lang = (cfg.engines.get(verifier) or {}).get("lang", "auto")

                        text, conf, lang_used, notes = _run_detector_only(img, lang)
                        log.info("ocr:stage_out", extra={"strategy": strategy, "rot": rot, "len": len(text or "")})
                        if text:
                            best_text, best_conf, best_lang, best_notes = text, conf, lang_used, f"{notes};rot={rot}"

                    # if we found text for this rotation, we can break out for first_nonempty or two_stage
                    if best_text and strategy in {"first_nonempty", "two_stage"}:
                        break

            if best_text:
                found_w.writerow([fp.name, best_text, best_lang or "", best_conf if best_conf is not None else "", best_notes])
                found += 1
                _jsonl({"file": fp.name, "action": "found", "engine": best_notes, "text_len": len(best_text)})
            else:
                fail_w.writerow([fp.name, "empty"])
                failed += 1
                _jsonl({"file": fp.name, "action": "empty"})

            pm.set_counts(processed, found=found, review=0)

    finally:
        try:
            found_f.close()
            fail_f.close()
            log_f.close()
        except Exception:
            pass

    elapsed = time.time() - t0
    log.info("ocr:end", extra={
        "found": found,
        "failed": failed,
        "total": total,
        "secs": round(elapsed, 2),
        "out_dir": str(run_dir),
    })
    return {
        "found_csv": str(found_csv),
        "failures_csv": str(failures_csv),
        "jsonl": str(jsonl_log),
        "found": found,
        "failed": failed,
        "processed": processed,
        "failures": failed,
        "total": total,
        "out_dir": str(run_dir),
        "run_dir": str(run_dir),
        "secs": elapsed,
    }
