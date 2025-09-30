# src/fait/audio/pipelines/speaker_recognition_pipeline.py
from __future__ import annotations
import os, time, json, shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple
import numpy as np
import logging

from fait.core.paths import get_paths
from fait.core.utils import (
    ensure_folder, is_audio_file, file_md5, ProgressMeter,
    fuse_scores as fuse2,  # pairwise fusion helper
)
from fait.audio.services.speaker_recognition_service import get_speaker_service
from fait.audio.speaker_recognition.base import AudioEmbedder

log = logging.getLogger("fait.audio.pipelines.speaker_recognition")

# ---------- Config ----------
@dataclass
class SingleStageCfg:
    tau: float = 0.70  # accept if score >= tau

@dataclass
class TwoStageCfg:
    method: str = "weighted"   # "and"|"weighted"|"product"|"max"|"sum"|"logistic"
    alpha: float = 0.60        # used if method="weighted"
    tau_star: float = 0.70     # decision threshold for fused score

@dataclass
class ThreeStageCfg:
    method: str = "weighted"
    alpha12: float = 0.60      # fuse(primary, secondary)
    alpha123: float = 0.60     # fuse(fused12, tertiary)
    tau_star: float = 0.70

@dataclass
class SpeakerModelsCfg:
    speechbrain_id: str = "speechbrain/spkrec-ecapa-voxceleb"
    titanet_id:    str = "nvidia/speakerverification_en_titanet_large"
    wavlm_id:      str = "microsoft/wavlm-base-plus"

Strategy = Literal[
    "speechbrain_only",     # single-stage ECAPA
    "titanet_only",         # single-stage TitaNet-L
    "detector_only",        # Titanet or WavLM (choose one as 'detector')
    "two_stage",            # SpeechBrain + (Titanet or WavLM)
    "three_stage",          # SpeechBrain + Titanet + WavLM
    "auto"
]

@dataclass
class SpeakerMatchConfig:
    # I/O (quickstart sets these)
    reference_dir: str
    gallery_dir: str
    output_dir: Optional[str] = None
    save_pairs: bool = False

    # Strategy knobs
    strategy: Strategy = "auto"
    detector: Literal["titanet", "wavlm", "auto", "none"] = "auto"
    tertiary: Literal["titanet", "wavlm", "auto", "none"] = "auto"

    # thresholds / fusion
    single: SingleStageCfg = SingleStageCfg()
    two_stage: TwoStageCfg = TwoStageCfg()
    three_stage: ThreeStageCfg = ThreeStageCfg()

    # models
    models: SpeakerModelsCfg = SpeakerModelsCfg()

    # progress
    progress: Literal["log","none"] = "log"
    progress_every: int = 10

    # cache usage
    use_cache: bool = True

# ---------- Helpers ----------
def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32, copy=False); b = b.astype(np.float32, copy=False)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(np.dot(a, b) / (na * nb))

def _resolve_strategy(cfg: SpeakerMatchConfig) -> Tuple[str, str, str]:
    """
    Return (strategy, detector, tertiary) after applying:
    - Only honor .env when strategy, detector, tertiary are ALL 'auto' in YAML.
    - For single-stage (speechbrain_only|titanet_only), force detector='none', tertiary='none'.
    """
    s = (cfg.strategy  or "auto").strip().lower()
    d = (cfg.detector  or "auto").strip().lower()
    t = (cfg.tertiary  or "auto").strip().lower()

    if s == d == t == "auto":
        env_s = os.getenv("FAIT_AUDIO_STRATEGY", "").strip().lower()
        env_d = os.getenv("FAIT_AUDIO_DETECTOR", "").strip().lower()
        env_t = os.getenv("FAIT_AUDIO_TERTIARY", "").strip().lower()
        s = env_s or "two_stage"
        d = env_d or "titanet"
        t = env_t or ("wavlm" if s == "three_stage" and d == "titanet"
                      else "titanet" if s == "three_stage" and d == "wavlm"
                      else "none")
    else:
        if s == "auto": s = "two_stage"
        if s in {"speechbrain_only", "titanet_only"}:
            d, t = "none", "none"
        elif s == "detector_only":
            if d == "auto": d = "titanet"
            t = "none"
        elif s == "two_stage":
            if d == "auto": d = "titanet"
            t = "none"
        elif s == "three_stage":
            if d == "auto": d = "titanet"
            if t == "auto":
                t = "wavlm" if d == "titanet" else "titanet"
        else:
            s, d, t = "two_stage", "titanet", "none"

    if s in {"speechbrain_only","titanet_only"}:
        d, t = "none", "none"
    if s in {"detector_only", "two_stage"} and d not in {"titanet", "wavlm"}:
        d = "titanet"
    if s == "three_stage" and t not in {"titanet", "wavlm"}:
        t = "wavlm" if d == "titanet" else "titanet"
    return s, d, t

def _fuse_two(a: float, b: float, cfg: TwoStageCfg) -> Tuple[float,float,bool]:
    from types import SimpleNamespace
    fcfg = SimpleNamespace(method=cfg.method, alpha=getattr(cfg, "alpha", 0.6), tau_star=cfg.tau_star)
    fused, tau, ok = fuse2(a, b, "speaker", fcfg)
    return float(fused), float(tau), bool(ok)

def _fuse_three(a: float, b: float, c: float, cfg: ThreeStageCfg) -> Tuple[float,float,bool]:
    from types import SimpleNamespace
    f1 = SimpleNamespace(method=cfg.method, alpha=cfg.alpha12,  tau_star=cfg.tau_star)
    s12, _t1, _ok1 = fuse2(a, b, "speaker", f1)
    f2 = SimpleNamespace(method=cfg.method, alpha=cfg.alpha123, tau_star=cfg.tau_star)
    fused, tau2, ok2 = fuse2(float(s12), c, "speaker", f2)
    return float(fused), float(tau2), bool(ok2)

# ---------- Pipeline ----------
def run_speaker_match(cfg: SpeakerMatchConfig) -> Dict:
    t0 = time.time()
    paths = get_paths()
    base = Path(cfg.output_dir) if cfg.output_dir else (paths.outputs / "audio" / "speaker_recognition")
    ensure_folder(base)

    strategy, detector, tertiary = _resolve_strategy(cfg)
    out_dir = base / f"{strategy}_{int(t0)}"
    found_dir = out_dir / "found_audio"
    ensure_folder(out_dir)

    svc = get_speaker_service()

    # Build embedders per strategy
    sb = tn = wl = None
    if strategy in {"speechbrain_only", "two_stage", "three_stage"}:
        sb = svc.get_speechbrain(cfg.models.speechbrain_id)
    if strategy == "titanet_only":
        tn = svc.get_titanet(cfg.models.titanet_id)

    if strategy in {"detector_only", "two_stage", "three_stage"} and detector != "none":
        if detector == "titanet":
            tn = tn or svc.get_titanet(cfg.models.titanet_id)
        elif detector == "wavlm":
            wl = wl or svc.get_wavlm(cfg.models.wavlm_id)

    if strategy == "three_stage" and tertiary != "none":
        if tertiary == "titanet" and tn is None: tn = svc.get_titanet(cfg.models.titanet_id)
        if tertiary == "wavlm"   and wl is None: wl = svc.get_wavlm(cfg.models.wavlm_id)

    # Reference centroid(s)
    ref_dir = Path(cfg.reference_dir); gal_dir = Path(cfg.gallery_dir)
    assert ref_dir.is_dir(), f"reference_dir not found: {ref_dir}"
    assert gal_dir.is_dir(), f"gallery_dir not found: {gal_dir}"

    def _mean(embedder: AudioEmbedder | None) -> Optional[np.ndarray]:
        if embedder is None: return None
        return embedder.mean_embedding_from_folder(str(ref_dir), use_cache=cfg.use_cache)

    ref_sb = _mean(sb)
    ref_tn = _mean(tn)
    ref_wl = _mean(wl)

    # Iterate gallery
    files = [p for p in gal_dir.iterdir() if p.is_file() and is_audio_file(p)]
    total = len(files)
    pm = ProgressMeter(total=total, label="speaker_match:progress", logger=log, emit_every_n=5, emit_every_sec=2.0)

    processed = found = 0
    log_jsonl = out_dir / "log.jsonl"
    with log_jsonl.open("w", encoding="utf-8") as jf:
        for f in files:
            processed += 1

            def _sim(ref: Optional[np.ndarray], emb: Optional[np.ndarray]) -> Optional[float]:
                if ref is None or emb is None: return None
                return 0.5 * (_cosine(ref, emb) + 1.0)  # [-1,1] -> [0,1]

            e_sb = sb.embed_file(str(f), use_cache=cfg.use_cache) if sb else None
            e_tn = tn.embed_file(str(f), use_cache=cfg.use_cache) if tn else None
            e_wl = wl.embed_file(str(f), use_cache=cfg.use_cache) if wl else None

            s_sb = _sim(ref_sb, e_sb) if sb else None
            s_tn = _sim(ref_tn, e_tn) if tn else None
            s_wl = _sim(ref_wl, e_wl) if wl else None

            accepted = False
            fused_score = None
            threshold = None

            if strategy == "speechbrain_only":
                fused_score = s_sb or 0.0
                threshold   = cfg.single.tau
                accepted    = fused_score >= threshold

            elif strategy == "titanet_only":
                fused_score = s_tn or 0.0
                threshold   = cfg.single.tau
                accepted    = fused_score >= threshold

            elif strategy == "detector_only":
                s_det = s_tn if detector == "titanet" else s_wl
                fused_score = s_det or 0.0
                threshold   = cfg.two_stage.tau_star
                accepted    = fused_score >= threshold

            elif strategy == "two_stage":
                s_det = s_tn if detector == "titanet" else s_wl
                a, b = (s_sb or 0.0), (s_det or 0.0)
                fused_score, threshold, accepted = _fuse_two(a, b, cfg.two_stage)

            elif strategy == "three_stage":
                s_det = s_tn if detector == "titanet" else s_wl
                s_ter = s_wl if detector == "titanet" else s_tn
                a, b, c = (s_sb or 0.0), (s_det or 0.0), (s_ter or 0.0)
                fused_score, threshold, accepted = _fuse_three(a, b, c, cfg.three_stage)

            entry = {
                "file": str(f),
                "hash": file_md5(f),
                "strategy": strategy,
                "detector": detector,
                "tertiary": tertiary,
                "scores": {
                    "speechbrain": s_sb,
                    "titanet": s_tn,
                    "wavlm": s_wl,
                    "fused": fused_score,
                },
                "threshold": threshold,
                "accepted": bool(accepted),
            }
            jf.write(json.dumps(entry) + "\n"); jf.flush()

            if accepted:
                ensure_folder(found_dir)
                shutil.copy2(f, found_dir / f.name)
                found += 1

            pm.set_counts(processed, found=found, review=0)

    pm.close()
    dur = time.time() - t0
    log.info("speaker_match:done", extra={"processed": processed, "found": found, "secs": round(dur, 2)})

    return {
        "processed": processed,
        "found": found,
        "run_dir": str(out_dir),
        "log_jsonl": str(log_jsonl),
    }
