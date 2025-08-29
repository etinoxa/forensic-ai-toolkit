# src/fait/core/paths.py
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    """
    Heuristically find the repo root:
    - Walk up from this file until we find a directory containing 'src'.
    - Fallback to 4 levels up (…/src/fait/core/paths.py -> parents[3]).
    - Final fallback: current working directory.
    """
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "src").exists():
            return p
    try:
        return here.parents[3]
    except IndexError:
        return Path.cwd()


def _default_home() -> Path:
    # Default FAIT home lives under the repo root
    return _repo_root() / ".fait"


def _env_path(name: str, default: Path) -> Path:
    """
    Read a path from env; treat missing or blank values as 'unset'.
    Always expanduser() and resolve() to avoid surprises.
    """
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return Path(v).expanduser().resolve()


@dataclass(frozen=True)
class FaitPaths:
    repo_root: Path
    fait_root: Path
    cache_root: Path
    models_cache: Path
    models_face_match: Path
    models_object_screen: Path
    models_llm: Path
    logs: Path
    outputs: Path
    embedding_cache: Path


_paths_singleton: FaitPaths | None = None


def get_paths() -> FaitPaths:
    root = _repo_root()
    fait_root = root / ".fait"
    cache_root = fait_root / "cache"
    models_cache = cache_root / "models"
    return FaitPaths(
        repo_root=root,
        fait_root=fait_root,
        cache_root=cache_root,
        models_cache=models_cache,
        models_face_match=models_cache / "face_match",
        models_object_screen=models_cache / "object_screen",
        models_llm=models_cache / "llm",
        logs=Path(os.getenv("FAIT_LOGS_DIR", str(fait_root / "logs"))),
        outputs=Path(os.getenv("FAIT_OUTPUTS_DIR", str(fait_root / "outputs"))),
        embedding_cache=Path(os.getenv("FAIT_EMBEDDING_CACHE", str(cache_root / "embedding_cache"))),
    )


def reset_paths_for_tests() -> None:
    """
    Clear the cached singleton. Useful for tests that mutate env vars
    and need get_paths() to re-resolve directories.
    """
    global _paths_singleton
    _paths_singleton = None

def ensure_on_first_write(p: Path) -> None:
    # call this only when you’re about to write; no-op otherwise
    p.mkdir(parents=True, exist_ok=True)
