from __future__ import annotations
import os, shutil, time
from datetime import datetime
from typing import List, Dict, Tuple

import logging
from fait.core.interfaces import Embedder
from fait.core.paths import get_paths
from fait.core.utils import ensure_folder, is_image_file, compute_distance, write_report, plot_distances, sort_pairs, to_safe_filename

log = logging.getLogger("fait.vision.pipelines.face_match")

def run_face_match(
    embedder,
    reference_dir: str,
    gallery_dir: str,
    output_dir: str | None = None,     # ← optional
    thresholds: list[float] = (),
    metric: str = "euclidean",
    use_cache: bool = True,
    plot_results: bool = False,
    topk_preview: int = 10,
):
    # resolve default outputs if not provided
    if output_dir is None:
        base = get_paths().outputs / "face"
        ensure_folder(base)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tag = to_safe_filename(embedder.name())
        output_dir = str(base / f"{tag}_{stamp}")

    # stable context for all logs
    ctx = {
        "component": embedder.name(),
        # "reference_dir": reference_dir,
        # "gallery_dir": gallery_dir,
        # "output_dir": output_dir,
        "thresholds": thresholds,
        "metric": metric,
    }
    log.info("face_match:start", extra=ctx)

    t0 = time.time()
    processed = 0
    counts = {t: 0 for t in thresholds}
    distances = []
    report_path = None
    plot_path = None

    try:
        ensure_folder(output_dir)

        # compute reference
        ref = embedder.mean_embedding(reference_dir, use_cache=use_cache)

        # iterate gallery
        files = [f for f in os.listdir(gallery_dir) if is_image_file(os.path.join(gallery_dir, f))]
        for fname in files:
            fp = os.path.join(gallery_dir, fname)
            emb = embedder.embed_image(fp, use_cache=use_cache)
            if emb is None:
                continue
            d = compute_distance(ref, emb, metric)
            distances.append((fname, float(d)))
            processed += 1
            for t in thresholds:
                ensure_folder(os.path.join(output_dir, f"threshold_{t}"))
                if d <= t:
                    ensure_folder(os.path.join(output_dir, f"threshold_{t}"))
                    shutil.copy(fp, os.path.join(output_dir, f"threshold_{t}", fname))
                    counts[t] += 1

        # artifacts
        report_path = write_report(
            output_dir=output_dir,
            model_type=embedder.name(),
            metric_name=metric,
            processed=processed,
            elapsed_seconds=time.time() - t0,
            match_counts=counts,
            pairs=distances,
            top_k=topk_preview,
            higher_is_better=False,
            filename="matching_report.txt",
        )
        if plot_results and distances:
            safe_tag = to_safe_filename(embedder.name())
            plot_path = plot_distances(
                distances=distances,
                thresholds=thresholds,
                output_dir=output_dir,
                metric_name=metric,
                filename=f"{safe_tag}_{metric}.png",
                top_n=20,
            )

        elapsed_ms = int((time.time() - t0) * 1000)
        log.info(
            "face_match:done",
            extra={**ctx, "processed": processed, "elapsed_ms": elapsed_ms,
                   "matches_per_threshold": counts, "report_path": report_path,
                   "plot_path": plot_path}
        )

        return {
            "model": embedder.name(),
            "metric": metric,
            "processed": processed,
            "elapsed_seconds": elapsed_ms / 1000.0,
            "matches_per_threshold": counts,
            "closest": sort_pairs(distances, higher_is_better=False)[:topk_preview],
            "report_path": report_path,
            "plot_path": plot_path,
        }

    except Exception:
        elapsed_ms = int((time.time() - t0) * 1000)
        # include partial progress so failures are diagnosable
        log.exception("face_match:error", extra={**ctx, "processed": processed, "elapsed_ms": elapsed_ms})
        raise