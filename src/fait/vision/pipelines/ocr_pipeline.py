# src/fait/vision/pipelines/ocr_pipeline.py
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

from fait.core.paths import get_paths
from fait.core.utils import (
    ensure_folder,
    is_image_file,
    human_size,
    ProgressMeter,
)
# One true config types & loader
from fait.vision.ocr.config import OcrConfig, FusionCfg, load_ocr_config, EngineCfg
# Engine factory (do NOT import the module named `engines` to avoid name clashes)
from fait.vision.ocr.engines import get_engine



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


def _sanitize_engine_order(cfg: OcrConfig) -> List[str]:
    """
    Keep only enabled engines, preserve the user-specified order.
    cfg.engines is normalized to dicts later so we can .get('enabled', True).
    """
    enabled = {
        name for name, e in (cfg.engines or {}).items()
        if (isinstance(e, dict) and e.get("enabled", True)) or (not isinstance(e, dict) and getattr(e, "enabled", True))
    }
    return [name for name in (cfg.engine_order or []) if name in enabled]


def _resolve_strategy_verifier(fu: FusionCfg) -> Tuple[str, str]:
    strategy = (fu.strategy or "first_nonempty").lower()
    verifier = (fu.verifier or "none").lower()

    env_s = (os.getenv("OCR_STRATEGY") or os.getenv("FAIT_OCR_STRATEGY") or "").strip().lower()
    env_v = (os.getenv("OCR_VERIFIER") or os.getenv("FAIT_OCR_VERIFIER") or "").strip().lower()

    allowed_strats = {
        "first_nonempty","best_of","consensus","two_stage",
        "tesseract_only","paddle_only","trocr_only","donut_only","doctr_only",
    }
    allowed_ver = {"trocr","donut","tesseract","paddleocr","doctr","none"}

    # ENV ALWAYS WINS if valid
    if env_s in allowed_strats:
        strategy = env_s
    if env_v in allowed_ver:
        verifier = env_v

    if strategy != "two_stage":
        verifier = "none"
    return strategy, verifier


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
        if w < min_dim or h < min_dim:
            return True, f"too_small ({w}x{h} < {min_dim}px)"
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

    # Resolve dirs
    gallery_dir = Path(cfg.gallery_dir) if cfg.gallery_dir else Path(
        os.getenv("OCR_GALLERY_DIR", paths.repo_root / "datasets/images/text_ocr")
    )
    if not gallery_dir.exists():
        raise FileNotFoundError(f"Gallery not found: {gallery_dir}")

    base_out = Path(cfg.output_dir) if cfg.output_dir else (paths.outputs / "ocr" / _now_tag())
    run_dir = base_out
    ensure_folder(run_dir)

    # Fixed filenames here
    found_csv = run_dir / "found.csv"
    failures_csv = run_dir / "failures.csv"
    jsonl_log = run_dir / "log.jsonl"

    # Strategy/verifier resolution (ENV wins only when YAML=auto)
    strategy, verifier = _resolve_strategy_verifier(cfg.fusion)


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

    # Preserve the resolved value to decide pruning
    resolved_strategy = strategy

    # -------- engine order (enabled only) --------
    order = _sanitize_engine_order(cfg)

    # If *_only selected (use the RESOLVED strategy), prune BEFORE any construction
    if resolved_strategy in _ONLY_MAP:
        only = _ONLY_MAP[resolved_strategy]
        if only not in cfg.engines:
            for k in cfg.engines:
                if k.lower().startswith(only):
                    only = k
                    break
        order = [only]
        verifier = "none"  # any *_only ignores verifier
        log.info("ocr:engine_order_resolved", extra={"order": order})

    if not order:
        raise RuntimeError("No OCR engines available/enabled after config filtering.")

    # Now normalize the runtime policy after pruning
    if resolved_strategy in _ONLY_MAP:
        strategy = "first_nonempty"

    # --------- LAZY engine getter: create an engine only when it is used ---------
    _engines_cache: Dict[str, object] = {}

    def _eng(name: str):
        if name not in _engines_cache:
            _engines_cache[name] = get_engine(name, cfg.engines.get(name))
        return _engines_cache[name]

    # IO setup
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
                            log.info("ocr:call", extra={"engine": name, "rot": rot, "lang": lang})
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
                        if not order:
                            continue
                        primary_name = order[0]
                        p_lang = cfg.engines[primary_name].get("lang", "auto")
                        p_text, p_conf = _normalize_engine_result(_eng(primary_name).ocr(img, lang=p_lang))
                        best_text, best_conf, best_notes, best_lang = p_text, p_conf, f"{primary_name};rot={rot}", p_lang

                        if not p_text or (p_conf is not None and p_conf < 0.5):
                            if verifier != "none":
                                v_lang = cfg.engines.get(verifier, {}).get("lang", "auto")
                                v_text, v_conf = _normalize_engine_result(_eng(verifier).ocr(img, lang=v_lang))
                                if v_text and (p_conf is None or (v_conf is not None and v_conf >= p_conf)):
                                    best_text, best_conf, best_notes, best_lang = v_text, v_conf, f"{verifier};rot={rot}", v_lang

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
        "secs": elapsed,
    }
