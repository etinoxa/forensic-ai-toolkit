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
    home: Path
    cache: Path
    models_cache: Path
    embeddings_cache: Path
    outputs: Path
    logs: Path
    tmp: Path


_paths_singleton: FaitPaths | None = None


def get_paths() -> FaitPaths:
    """
    Central place to resolve FAIT directories.
    Honors (if set and non-blank):
      FAIT_HOME, FAIT_CACHE_DIR, FAIT_MODELS_CACHE_DIR,
      FAIT_EMBEDDINGS_CACHE_DIR, FAIT_OUTPUTS_DIR,
      FAIT_LOGS_DIR, FAIT_TMP_DIR
    """
    global _paths_singleton
    if _paths_singleton is not None:
        return _paths_singleton

    home = _env_path("FAIT_HOME", _default_home()).resolve()
    cache = _env_path("FAIT_CACHE_DIR", home / "cache")
    models = _env_path("FAIT_MODELS_CACHE_DIR", cache / "models")
    embeds = _env_path("FAIT_EMBEDDINGS_CACHE_DIR", cache / "embeddings")
    outputs = _env_path("FAIT_OUTPUTS_DIR", home / "outputs")
    logs = _env_path("FAIT_LOGS_DIR", home / "logs")
    tmp = _env_path("FAIT_TMP_DIR", home / "tmp")

    _paths_singleton = FaitPaths(home, cache, models, embeds, outputs, logs, tmp)
    return _paths_singleton


def reset_paths_for_tests() -> None:
    """
    Clear the cached singleton. Useful for tests that mutate env vars
    and need get_paths() to re-resolve directories.
    """
    global _paths_singleton
    _paths_singleton = None
