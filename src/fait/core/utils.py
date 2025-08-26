# src/fait/core/utils.py
from __future__ import annotations
import os, re, io, json, hashlib, pickle
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Optional, Any

import numpy as np


# ───────────────────────────── Filesystem & IO ─────────────────────────────

def ensure_folder(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def is_image_file(path: str | Path) -> bool:
    p = str(path).lower()
    return os.path.isfile(path) and p.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))

def is_video_file(path: str | Path) -> bool:
    p = str(path).lower()
    return os.path.isfile(path) and p.endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))

def walk_files(root: str | Path,
               allowed_exts: Optional[Iterable[str]] = None,
               max_depth: Optional[int] = None) -> Iterator[Path]:
    """
    Yield all files under `root`. If `allowed_exts` is given, filter by extension.
    If `max_depth` is set, limit traversal depth (0 => only root).
    """
    root = Path(root)
    if root.is_file():
        if not allowed_exts or root.suffix.lower() in {e.lower() for e in allowed_exts}:
            yield root
        return

    start_depth = len(root.parts)
    for dirpath, _dirnames, filenames in os.walk(root):
        if max_depth is not None and (len(Path(dirpath).parts) - start_depth) > max_depth:
            continue
        for fn in filenames:
            p = Path(dirpath) / fn
            if allowed_exts and p.suffix.lower() not in {e.lower() for e in allowed_exts}:
                continue
            yield p

def to_safe_filename(name: str, maxlen: int = 120) -> str:
    s = re.sub(r"[^\w\-.]+", "_", name.strip())
    if len(s) > maxlen:
        root, ext = os.path.splitext(s)
        s = root[: maxlen - len(ext) - 1] + "_" + ext
    return s

def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_folder(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ───────────────────────────── Hashing / Provenance ─────────────────────────────

def sha256_file(path: str | Path, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(bufsize)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_md5(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Fast MD5 of a file in chunks (1 MiB default).
    Returns the hex digest string. Safe for large files.
    """
    p = Path(path)
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()

# ───────────────────────────── Embedding Cache ─────────────────────────────

def cache_path(embed_cache_dir: str | Path, image_path: str | Path, key_suffix: str) -> str:
    """
    Return a stable base path (without extension) for caching an embedding.
    We append .pkl (default) or .npy in save/load helpers.
    Example: "<embed_cache_dir>/<md5>_<filename>.emb"
    """
    ensure_folder(embed_cache_dir)
    image_path = str(image_path)
    digest = hashlib.md5(f"{image_path}_{key_suffix}".encode()).hexdigest()
    base = os.path.basename(image_path)
    return os.path.join(str(embed_cache_dir), f"{digest}_{base}.emb")

def save_embedding(path: str | Path, emb: np.ndarray) -> str:
    """
    Save embedding; if `path` ends with .npy we use np.save, otherwise pickle (.pkl).
    Returns the actual written path (extension may be added).
    """
    p = Path(str(path))
    if p.suffix.lower() == ".npy":
        ensure_parent_dir(p)
        np.save(p, emb.astype(np.float32, copy=False))
        return str(p)

    # default to .pkl if no extension
    if p.suffix == "":
        p = p.with_suffix(".pkl")

    ensure_parent_dir(p)
    with open(p, "wb") as f:
        pickle.dump(emb, f, protocol=pickle.HIGHEST_PROTOCOL)
    return str(p)

def load_embedding(path: str | Path) -> np.ndarray:
    path = str(path)
    _, ext = os.path.splitext(path)
    if ext.lower() == ".npy":
        return np.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


# ───────────────────────────── Math / Metrics ─────────────────────────────

def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < eps else (v / n)

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    """
    Cosine similarity in [-1, 1]; works with non-normalized inputs.
    """
    an = float(np.linalg.norm(a)); bn = float(np.linalg.norm(b))
    if an < eps or bn < eps:
        return 0.0
    return float(np.dot(a, b) / (an * bn))

def compute_distance(a: np.ndarray, b: np.ndarray, metric: str = "euclidean") -> float:
    """
    Distance between vectors:
      - 'euclidean' : L2 distance
      - 'cosine'    : 1 - cosine_similarity (smaller is better)
    """
    if metric == "euclidean":
        return float(np.linalg.norm(a - b))
    if metric == "cosine":
        return 1.0 - cosine_similarity(a, b)
    raise ValueError(f"Unknown metric: {metric}")

def sort_pairs(pairs: List[Tuple[str, float]], higher_is_better: bool) -> List[Tuple[str, float]]:
    return sorted(pairs, key=lambda x: x[1], reverse=higher_is_better)

def topk_pairs(pairs: List[Tuple[str, float]], k: int, higher_is_better: bool) -> List[Tuple[str, float]]:
    return sort_pairs(pairs, higher_is_better)[:k]


# ───────────────────────────── Plotting (headless-safe) ─────────────────────────────

def _use_headless_backend() -> None:
    # Safe to call multiple times; avoids GUI backends on servers.
    import matplotlib
    matplotlib.use("Agg")

def plot_scores(
    pairs: List[Tuple[str, float]],
    thresholds: Iterable[float],
    output_dir: str | Path,
    filename: str,
    xlabel: str,
    title: str,
    higher_is_better: bool,
    top_n: int = 20,
) -> Optional[str]:
    """
    Generic horizontal bar plot for either distances or similarities.
    Does nothing (returns None) if `pairs` is empty.
    """
    if not pairs:
        return None

    _use_headless_backend()
    import matplotlib.pyplot as plt

    ensure_folder(output_dir)
    ranked = sort_pairs(pairs, higher_is_better=higher_is_better)[:top_n]
    labels = [x[0] for x in ranked]
    values = [x[1] for x in ranked]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(labels)), values)  # avoid specifying colors/styles explicitly
    for t in thresholds:
        plt.axvline(x=t, linestyle="--", alpha=0.7, label=f"Threshold {t}")
    plt.yticks(range(len(labels)), labels)
    plt.xlabel(xlabel)
    plt.title(title)
    try:
        # Only show legend if thresholds iterable is non-empty
        thresholds = list(thresholds)  # may be generator
        if thresholds:
            plt.legend()
    except Exception:
        pass
    plt.tight_layout()

    plot_path = str(Path(output_dir) / filename)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    return plot_path

def plot_distances(
    distances: List[Tuple[str, float]],
    thresholds: Iterable[float],
    output_dir: str | Path,
    metric_name: str = "euclidean",
    filename: Optional[str] = None,
    top_n: int = 20,
) -> Optional[str]:
    fname = filename or f"distance_plot_{metric_name}.png"
    return plot_scores(
        distances,
        thresholds,
        output_dir,
        fname,
        xlabel=f"{metric_name.title()} Distance",
        title=f"Distance Analysis (Top {top_n} closest)",
        higher_is_better=False,
        top_n=top_n,
    )

def plot_similarities(
    similarities: List[Tuple[str, float]],
    thresholds: Iterable[float],
    output_dir: str | Path,
    filename: str = "similarity_plot.png",
    top_n: int = 20,
) -> Optional[str]:
    return plot_scores(
        similarities,
        thresholds,
        output_dir,
        filename,
        xlabel="Cosine Similarity",
        title=f"Similarity Analysis (Top {top_n})",
        higher_is_better=True,
        top_n=top_n,
    )


# ───────────────────────────── Reporting ─────────────────────────────

def write_report(
    output_dir: str | Path,
    model_type: str,
    metric_name: str,
    processed: int,
    elapsed_seconds: float,
    match_counts: Dict[float, int],
    pairs: List[Tuple[str, float]],
    top_k: int = 10,
    higher_is_better: bool = False,
    filename: str = "matching_report.txt",
) -> str:
    """
    Generic matching report writer.
    `pairs` are (filename, score) tuples. For similarities set `higher_is_better=True`.
    """
    ensure_folder(output_dir)
    path = str(Path(output_dir) / filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write("=== FINAL REPORT ===\n")
        f.write(f"Model type: {model_type}\n")
        f.write(f"Metric: {metric_name}\n")
        f.write(f"Total gallery images processed: {processed}\n")
        f.write(f"Processing time: {elapsed_seconds:.2f} seconds\n\n")

        f.write("Matches per threshold:\n")
        for t in sorted(match_counts.keys(), reverse=higher_is_better):
            f.write(f"  Threshold {t}: {match_counts[t]} matches\n")

        if pairs:
            ranked = sort_pairs(pairs, higher_is_better=higher_is_better)[:top_k]
            f.write(f"\nTop {len(ranked)} {'best' if higher_is_better else 'closest'}:\n")
            for i, (nm, val) in enumerate(ranked, 1):
                f.write(f"  {i:2d}. {nm:<30} ({metric_name}: {val:.4f})\n")

        f.write("=" * 50 + "\n")

    return path


# ───────────────────────────── Exports ─────────────────────────────

__all__ = [
    # IO & FS
    "ensure_folder", "is_image_file", "is_video_file", "walk_files",
    "to_safe_filename", "write_jsonl",
    # Hashing / cache
    "sha256_file", "cache_path", "save_embedding", "load_embedding", "file_md5",
    # Math / metrics
    "l2_normalize", "cosine_similarity", "compute_distance",
    "sort_pairs", "topk_pairs",
    # Plotting & reporting
    "plot_scores", "plot_distances", "plot_similarities", "write_report",
]
