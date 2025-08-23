from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass

_DEF_HOME_NAME = ".fait"  # lives alongside your repo

def _find_repo_root(start: Path) -> Path | None:
    """Walk up to find a directory containing pyproject.toml or .git."""
    cur = start
    for _ in range(8):  # avoid walking to filesystem root forever
        if (cur / "pyproject.toml").exists() or (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def _default_home() -> Path:
    # 1) explicit override
    env_home = os.getenv("FAIT_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    # 2) repo-local .fait/
    here = Path(__file__).resolve()
    repo = _find_repo_root(here)
    if repo:
        return (repo / _DEF_HOME_NAME).resolve()
    # 3) fallback to user dir
    return (Path.home() / _DEF_HOME_NAME).resolve()

@dataclass(frozen=True)
class FaitPaths:
    home: Path
    cache: Path
    models_cache: Path
    embeddings_cache: Path
    outputs: Path
    logs: Path
    tmp: Path

    def ensure(self) -> "FaitPaths":
        for p in (self.home, self.cache, self.models_cache, self.embeddings_cache,
                  self.outputs, self.logs, self.tmp):
            p.mkdir(parents=True, exist_ok=True)
        return self

# singleton
_paths: FaitPaths | None = None

def _env_path(name: str, default: Path) -> Path:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return Path(v).expanduser().resolve()

def get_paths() -> FaitPaths:
    global _paths
    if _paths is not None:
        return _paths

    home = _default_home()

    # environment overrides for subdirs (optional)
    cache_dir = _env_path("FAIT_CACHE_DIR", home / "cache")
    models    = _env_path("FAIT_MODELS_CACHE_DIR", cache_dir / "models")
    embeds    = _env_path("FAIT_EMBEDDINGS_CACHE_DIR", cache_dir / "embeddings")
    outputs   = _env_path("FAIT_OUTPUTS_DIR", home / "outputs")
    logs      = _env_path("FAIT_LOGS_DIR", home / "logs")
    tmp       = _env_path("FAIT_TMP_DIR", home / "tmp")

    _paths = FaitPaths(
        home=home, cache=cache_dir, models_cache=models,
        embeddings_cache=embeds, outputs=outputs, logs=logs, tmp=tmp
    ).ensure()
    return _paths
