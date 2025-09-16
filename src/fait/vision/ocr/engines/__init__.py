# src/fait/vision/ocr/engines/__init__.py
from __future__ import annotations
from typing import Any, Dict, Optional

from .tesseract_engine import TesseractEngine
from .doctr_engine import DocTREngine
from .donut_engine import DonutEngine
from .trocr_engine import TrOCREngine
from .paddle_engine import PaddleEngine
# (Optionally re-export others when you add them)

__all__ = ["get_engine", "ENGINE_ALIASES"]

# Lazy import wrappers so optional deps don’t explode on import time.
def _import_tesseract():
    from .tesseract_engine import TesseractEngine
    return TesseractEngine

def _import_paddle():
    from .paddle_engine import PaddleEngine
    return PaddleEngine

def _import_trocr():
    from .trocr_engine import TrOCREngine
    return TrOCREngine

def _import_donut():
    from .donut_engine import DonutEngine
    return DonutEngine

def _import_doctr():
    from .doctr_engine import DocTREngine
    return DocTREngine


# Normalized name -> callable that returns the class
_ENGINE_LOADERS = {
    "tesseract": _import_tesseract,
    "paddleocr": _import_paddle,
    "paddle":    _import_paddle,      # alias
    "trocr":     _import_trocr,
    "donut":     _import_donut,
    "doctr":     _import_doctr,
}

# nice public list of valid keys / aliases
ENGINE_ALIASES = sorted(set(_ENGINE_LOADERS.keys()))


def _cfg_to_kwargs(cfg: Optional[Any]) -> Dict[str, Any]:
    """
    Accept None / dataclass / simple object / dict and produce kwargs for engine ctor.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    # dataclass or SimpleNamespace or object with public attributes
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}  # fallback


def get_engine(name: str, cfg: Optional[Any] = None):
    """
    Factory: returns an OCR engine instance by name.

    Parameters
    ----------
    name : str
        One of: 'tesseract', 'paddleocr' (or 'paddle'), 'trocr', 'donut', 'doctr'
    cfg : Optional[Any]
        A per-engine configuration object or dict; converted to kwargs.

    Raises
    ------
    ValueError if the engine name is unknown.
    ImportError if the engine module can’t be imported (missing dependency).
    """
    if not name:
        raise ValueError("get_engine: 'name' must be a non-empty string")

    key = str(name).strip().lower()
    if key not in _ENGINE_LOADERS:
        raise ValueError(
            f"Unknown OCR engine '{name}'. "
            f"Valid options: {', '.join(ENGINE_ALIASES)}"
        )

    try:
        engine_cls = _ENGINE_LOADERS[key]()  # lazy import + return class
    except ImportError as e:
        raise ImportError(
            f"Failed to import OCR engine '{name}'. "
            "Make sure its optional dependencies are installed."
        ) from e

    kwargs = _cfg_to_kwargs(cfg)
    # Only pass parameters accepted by the engine constructor
    try:
        import inspect
        sig = inspect.signature(engine_cls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        # If the engine supports a 'cache_dir' parameter and none provided, point it to .fait/cache/models/ocr
        if "cache_dir" in allowed and "cache_dir" not in kwargs:
            try:
                from fait.core.paths import get_paths
                from fait.core.utils import ensure_folder
                cache_root = get_paths().models_cache / "ocr"
                ensure_folder(cache_root)
                kwargs["cache_dir"] = str(cache_root)
            except Exception:
                # Non-fatal: if we can't resolve paths, proceed without cache_dir
                pass
    except Exception:
        # Best-effort filtering; if inspection fails, fall back to raw kwargs
        pass
    return engine_cls(**kwargs)