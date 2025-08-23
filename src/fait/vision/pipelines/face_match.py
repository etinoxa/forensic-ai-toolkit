from __future__ import annotations
import os, shutil, time
from datetime import datetime
from typing import List, Dict, Tuple

from fait.core.interfaces import Embedder
from fait.core.utils import ensure_folder, is_image_file, compute_distance, write_report, plot_distances, sort_pairs, to_safe_filename
from fait.core.paths import get_paths

def run_face_match(
    embedder: Embedder,
    reference_dir: str,
    gallery_dir: str,
    output_dir: str,
    thresholds: List[float],
    metric: str = "euclidean",   # for CLIP use "cosine" (i.e., 1 - cosine_sim)
    use_cache: bool = True,
    plot_results: bool = False,
    topk_preview: int = 10,
) -> Dict:
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = str(get_paths().outputs / "face" / f"run_{ts}")
    ensure_folder(output_dir)
    for t in thresholds: ensure_folder(os.path.join(output_dir, f"threshold_{t}"))

    start = time.time()
    ref = embedder.mean_embedding(reference_dir, use_cache=use_cache)

    distances: List[Tuple[str, float]] = []
    counts = {t: 0 for t in thresholds}
    files = [f for f in os.listdir(gallery_dir) if is_image_file(os.path.join(gallery_dir, f))]
    processed = 0

    for fname in files:
        fp = os.path.join(gallery_dir, fname)
        emb = embedder.embed_image(fp, use_cache=use_cache)
        if emb is None: continue
        d = compute_distance(ref, emb, metric)
        distances.append((fname, float(d))); processed += 1
        for t in thresholds:
            if d <= t:
                shutil.copy(fp, os.path.join(output_dir, f"threshold_{t}", fname))
                counts[t] += 1

    elapsed = time.time() - start
    report_path = write_report(output_dir, embedder.name(), metric, processed, elapsed, counts, distances, top_k=topk_preview)
    plot_path = None
    if plot_results and distances:
        safe_tag = to_safe_filename(embedder.name())  # strips /, (), commas, spaces, etc.
        plot_path = plot_distances(
            distances=distances,
            thresholds=thresholds,
            output_dir=output_dir,
            metric_name=metric,
            filename=f"{safe_tag}_{metric}.png",
            top_n=20,
        )

    return {
        "model": embedder.name(),
        "metric": metric,
        "processed": processed,
        "elapsed_seconds": elapsed,
        "matches_per_threshold": counts,
        "closest": sort_pairs(distances, False)[:topk_preview],
        "report_path": report_path,
        "plot_path": plot_path,
    }
