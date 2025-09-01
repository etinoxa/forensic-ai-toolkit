# src/fait/core/utils.py
from __future__ import annotations
from dataclasses import dataclass
import os, re, io, json, hashlib, pickle, math
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple, Dict, Optional, Any, Union, Mapping
import logging, json, time

import numpy as np


# ───────────────────────────── Filesystem & IO ─────────────────────────────


DEFAULT_AUDIO_EXTS = (
    ".wav", ".flac", ".mp3", ".m4a", ".aac",
    ".ogg", ".opus", ".wma", ".aiff", ".aif", ".aifc",
    ".amr", ".caf", ".mka", ".mp2", ".mpga"
)

DEFAULT_VIDEO_EXTS = (
    ".mp4", ".avi", ".mov", ".mkv", ".webm"
)

DEFAULT_IMAGE_EXTS = (
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
)
def ensure_folder(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def is_image_file(path: str | Path, exts: tuple[str, ...] = DEFAULT_IMAGE_EXTS) -> bool:
    p = str(path).lower()
    return Path(path).is_file() and p.endswith(exts)

def is_video_file(path: str | Path, exts: tuple[str, ...] = DEFAULT_VIDEO_EXTS) -> bool:
    p = str(path).lower()
    return Path(path).is_file() and p.endswith(exts)

def is_audio_file(path: str | Path, exts: tuple[str, ...] = DEFAULT_AUDIO_EXTS) -> bool:
    p = str(path).lower()
    return Path(path).is_file() and p.endswith(exts)


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

def append_jsonl(log_path: str | Path, obj: dict) -> None:
    """Append a JSON object to a .jsonl file (no persistent handle kept)."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")

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

def write_object_report(
    output_dir: Union[str, Path],
    strategy: str,
    device: str,
    prompts: List[str],
    models: Dict[str, str],
    thresholds: Dict[str, float],
    counts: Dict[str, int],
    found_files: List[str],
    filename: str = "report.txt",
) -> str:
    """Write a human-readable report for object screening and return the path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    with path.open("w", encoding="utf-8") as f:
        f.write("=== OBJECT SCREEN REPORT ===\n")
        f.write(f"Strategy      : {strategy}\n")
        f.write(f"Device        : {device}\n")
        f.write(f"Prompts       : {', '.join(prompts)}\n")

        # models
        for k, v in models.items():
            f.write(f"Model[{k}]    : {v}\n")

        # numeric thresholds
        for k, v in thresholds.items():
            f.write(f"{k:12}: {v}\n")

        f.write("\n")
        f.write(f"Processed     : {counts.get('processed', 0)}\n")
        f.write(f"Found images  : {counts.get('found', 0)}\n")
        f.write(f"Review queue  : {counts.get('review', 0)}\n")

        f.write("\nFound files:\n")
        if found_files:
            for nm in found_files:
                f.write(f"  - {nm}\n")
        else:
            f.write("  (none)\n")

    return str(path.resolve())

# ───────────────────────────── Progress Status ─────────────────────────────

@dataclass
class ProgressMeter:
    """
    Unifies progress logging for both face_match and object_screen.
    Prints a compact JSON payload in the message (easy to read/grep)
    and optionally writes a structured line to a JSONL log file.

    Example console line:
      object_screen:progress {"processed":20,"total":30,"found":4,"review":0,"imgs_per_s":1.18,"eta_s":8}
    """
    total: int
    label: str                        # e.g. "object_screen:progress" or "face_match:progress"
    logger: logging.Logger
    log_path: Optional[str | Path] = None
    emit_every_n: int = 5
    emit_every_sec: float = 2.0

    # internal state
    processed: int = 0
    found: int = 0
    review: int = 0
    _t0: float = 0.0
    _last_emit: float = 0.0

    def __post_init__(self) -> None:
        self._t0 = time.time()
        self._last_emit = 0.0
        # Emit an initial 0% line to show we're alive
        self._emit(force=True)

    def step(self, processed_inc: int = 1, found_inc: int = 0, review_inc: int = 0) -> None:
        """Increment counters and emit if thresholds are met."""
        self.processed += processed_inc
        if found_inc:
            self.found += found_inc
        if review_inc:
            self.review += review_inc
        self._maybe_emit()

    def set_counts(self, processed: int, found: Optional[int] = None, review: Optional[int] = None) -> None:
        """Directly set counters (if you manage them externally)."""
        self.processed = processed
        if found is not None:
            self.found = found
        if review is not None:
            self.review = review
        self._maybe_emit()

    def close(self) -> None:
        """Emit a final line."""
        self._emit(force=True)

    # --- internals ---
    def _maybe_emit(self) -> None:
        now = time.time()
        if (
            self.processed in (1, self.total) or
            (self.emit_every_n > 0 and self.processed % self.emit_every_n == 0) or
            (now - self._last_emit) >= self.emit_every_sec
        ):
            self._emit()
            self._last_emit = now

    def _emit(self, force: bool = False) -> None:
        elapsed = max(time.time() - self._t0, 1e-9)
        ips = self.processed / elapsed
        eta_s = int((self.total - self.processed) / ips) if ips > 1e-9 else None
        payload = {
            "processed": self.processed,
            "total": self.total,
            "found": self.found,
            "review": self.review,
            "imgs_per_s": round(ips, 2),
            "eta_s": eta_s,
        }
        # 1) Console (human friendly)
        # We embed compact JSON in the message so it shows cleanly regardless of formatter.
        self.logger.info("%s %s", self.label, json.dumps(payload, separators=(",", ":")))
        # 2) JSONL (machine friendly), if a path was provided
        if self.log_path:
            append_jsonl(self.log_path, {"event": "progress", **payload})


# ───────────────────────────── Fusion ─────────────────────────────

def _fcfg_get(cfg: Any, key: str, default: Any = None):
    """Read attribute or mapping key from a config-ish object."""
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return default

def _zscore(x: float, mean: Optional[float], std: Optional[float]) -> float:
    if mean is None or std is None or std <= 0:
        return x
    return (x - mean) / std

def _sigmoid(x: float) -> float:
    # numerically stable sigmoid
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)

def fuse_scores(
    gscore: float,
    cscore: float,
    label: str,
    fcfg: Any,
) -> tuple[float, float, bool]:
    """
    Generic score-level fusion.
    Returns: (fused_score, threshold_used, is_accept)

    Supports fcfg fields (attr or mapping):
      method: "and" | "weighted" | "sum" | "product" | "max" | "logistic"
      alpha, tau_star
      class_thresholds: dict
      gdino_only_default_tau
      gdino_mean, gdino_std, det_mean, det_std
      w_g, w_c, b   (for "logistic")
    """
    method = (_fcfg_get(fcfg, "method", "and") or "and").lower()
    alpha = float(_fcfg_get(fcfg, "alpha", 0.5))
    tau_star = float(_fcfg_get(fcfg, "tau_star", 0.6))
    class_thresholds = _fcfg_get(fcfg, "class_thresholds", {}) or {}
    default_tau = float(_fcfg_get(fcfg, "gdino_only_default_tau", 0.5))

    gdino_mean = _fcfg_get(fcfg, "gdino_mean", None)
    gdino_std  = _fcfg_get(fcfg, "gdino_std",  None)
    det_mean   = _fcfg_get(fcfg, "det_mean",   None)
    det_std    = _fcfg_get(fcfg, "det_std",    None)

    w_g = float(_fcfg_get(fcfg, "w_g", 1.0))
    w_c = float(_fcfg_get(fcfg, "w_c", 1.0))
    b   = float(_fcfg_get(fcfg, "b",   0.0))

    class_tau = float(class_thresholds.get(label.lower(), default_tau))
    tau = class_tau if method == "and" else tau_star

    # optional z-norm for methods that combine the two scores
    g = _zscore(gscore, gdino_mean, gdino_std)
    c = _zscore(cscore, det_mean,   det_std)

    if method == "and":
        fused = min(gscore, cscore)        # report the bottleneck score
        ok = (gscore >= class_tau) and (cscore >= class_tau)

    elif method == "weighted":
        fused = alpha * gscore + (1.0 - alpha) * cscore
        ok = fused >= tau

    elif method == "sum":
        fused = g + c
        ok = fused >= tau

    elif method == "product":
        fused = max(0.0, min(1.0, gscore * cscore))
        ok = fused >= tau

    elif method == "max":
        fused = max(gscore, cscore)
        ok = fused >= tau

    elif method == "logistic":
        fused = _sigmoid(w_g * g + w_c * c + b)
        ok = fused >= tau

    else:
        # fallback to weighted
        fused = alpha * gscore + (1.0 - alpha) * cscore
        ok = fused >= tau

    return float(fused), float(tau), bool(ok)


# ───────────────────────────── Exports ─────────────────────────────

__all__ = [
    # IO & FS
    "ensure_folder", "is_image_file", "is_video_file", "walk_files",
    "to_safe_filename", "write_jsonl", is_audio_file, "append_jsonl",
    # Hashing / cache
    "sha256_file", "cache_path", "save_embedding", "load_embedding", "file_md5",
    # Math / metrics
    "l2_normalize", "cosine_similarity", "compute_distance",
    "sort_pairs", "topk_pairs",
    # Plotting & reporting
    "plot_scores", "plot_distances", "plot_similarities", "write_report", "write_object_report",
    # Progress status
    "ProgressMeter",
    # Fusion
    "fuse_scores",
]
