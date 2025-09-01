# src/fait/audio/pipelines/speaker_match.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Literal
import os, time, shutil, logging
from pathlib import Path
import numpy as np

from fait.core.paths import get_paths
from fait.core.utils import (
    ensure_folder, is_audio_file, write_report, ProgressMeter
)
from ..services.speaker_service import get_speaker_service
from ..embeddings.base import AudioEmbedder

log = logging.getLogger("fait.audio.pipelines.speaker_match")

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))

@dataclass
class SpeakerMatchConfig:
    reference_dir: str
    gallery_dir: str
    thresholds: List[float]           # e.g., [0.70, 0.75, 0.80] (higher = stricter)
    embedder: Literal["speechbrain", "resemblyzer"] = "speechbrain"
    use_cache: bool = True
    plot_results: bool = False        # placeholder; add plot if you like

def _run_dir_name(model_name: str, device: str) -> str:
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_s = model_name.replace(" ", "_")
    return f"{model_s}__{device}__{ts}"

def run_speaker_match(cfg: SpeakerMatchConfig) -> Dict:
    t0 = time.time()
    paths = get_paths()

    # build embedder
    svc = get_speaker_service()
    if cfg.embedder == "speechbrain":
        emb: AudioEmbedder = svc.get_speechbrain()
    else:
        emb = svc.get_resemblyzer()

    device = "cuda" if "cuda" in getattr(emb, "device", "cpu") else "cpu"
    base = paths.outputs / "audio" / "speaker_match"
    ensure_folder(base)
    run_dir = base / _run_dir_name(emb.name(), device)
    ensure_folder(run_dir)

    log.info("speaker_match:start", extra={
        "embedder": emb.name(),
        "reference_dir": cfg.reference_dir,
        "gallery_dir": cfg.gallery_dir,
        "thresholds": cfg.thresholds,
    })

    # mean reference embedding
    ref = emb.mean_embedding_from_folder(cfg.reference_dir, use_cache=cfg.use_cache)

    # iterate gallery
    files = [f for f in os.listdir(cfg.gallery_dir)
             if is_audio_file(os.path.join(cfg.gallery_dir, f))]
    distances: List[Tuple[str, float]] = []  # store as "1 - similarity" for compatibility
    sims: List[Tuple[str, float]] = []
    match_counts = {float(t): 0 for t in cfg.thresholds}
    processed = 0

    pm = ProgressMeter(total=len(files), label="speaker_match:progress", logger=log,
                       emit_every_n=5, emit_every_sec=2.0)

    for fname in files:
        fpath = os.path.join(cfg.gallery_dir, fname)
        g = emb.embed_file(fpath, use_cache=cfg.use_cache)
        if g is None:
            continue
        s = cosine_sim(ref, g)
        sims.append((fname, s))
        distances.append((fname, 1.0 - s))
        processed += 1
        pm.set_counts(processed, found=sum(match_counts.values()), review=0)

        for t in cfg.thresholds:
            if s >= t:
                td = run_dir / f"threshold_{t}"
                ensure_folder(td)
                shutil.copy2(fpath, td / fname)
                match_counts[float(t)] += 1

    pm.close()
    dt = time.time() - t0

    # write report (similarities, higher is better)
    report_path = write_report(
        output_dir=run_dir,
        model_type=emb.name(),
        metric_name="Cosine Similarity",
        processed=processed,
        elapsed_seconds=dt,
        match_counts=match_counts,
        pairs=sims,
        higher_is_better=True,
        filename="speaker_matching_report.txt",
    )

    log.info("speaker_match:done", extra={
        "embedder": emb.name(),
        "processed": processed,
        "run_dir": str(run_dir),
        "report_path": report_path
    })

    return {
        "processed": processed,
        "run_dir": str(run_dir),
        "report_path": report_path,
        "matches_per_threshold": match_counts,
        "top": sorted(sims, key=lambda x: -x[1])[:10],
    }
