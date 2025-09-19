# src/fait/vision/ocr/engines/__init__.py
from __future__ import annotations
from typing import Any, Dict, Optional, Callable

# Keep these imports lightweight; real classes are imported inside loader funcs
# so optional deps don't explode at import time.
__all__ = ["get_engine", "ENGINE_ALIASES"]

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

# Normalized name -> callable that returns the class (still lazy)
_ENGINE_LOADERS: Dict[str, Callable[[], type]] = {
    "tesseract": _import_tesseract,
    "paddleocr": _import_paddle,
    "paddle":    _import_paddle,   # alias
    "trocr":     _import_trocr,
    "donut":     _import_donut,
    "doctr":     _import_doctr,
}

ENGINE_ALIASES = sorted(set(_ENGINE_LOADERS.keys()))

def _cfg_to_kwargs(cfg: Optional[Any]) -> Dict[str, Any]:
    """
    Accept None / dataclass / simple object / dict and produce kwargs for engine ctor.
    """
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "__dict__"):       # dataclass / SimpleNamespace / object
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}

class _LazyEngine:
    def __init__(self, engine_cls, kwargs):
        self._engine_cls = engine_cls
        self._kwargs = kwargs
        self._inst = None

    def _ensure(self):
        if self._inst is None:
            self._inst = self._engine_cls(**self._kwargs)

    def _call(self, method, *args, **kwargs):
        import inspect
        self._ensure()
        # Try requested method first
        fn = getattr(self._inst, method, None)
        if callable(fn):
            try:
                sig = inspect.signature(fn)
                # keep only kwargs that the method accepts
                filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
            except Exception:
                filtered = kwargs
            return fn(*args, **filtered)

        # Fallback: engines that only implement .recognize(img)
        if method == "ocr" and hasattr(self._inst, "recognize"):
            fn = getattr(self._inst, "recognize")
            # recognize usually only takes (img); ignore extra kwargs like lang
            if callable(fn):
                return fn(*args)
        raise AttributeError(f"{self._inst.__class__.__name__} has no method '{method}'")

    def ocr(self, *args, **kwargs):
        return self._call("ocr", *args, **kwargs)

    def recognize(self, *args, **kwargs):
        return self._call("recognize", *args, **kwargs)

    def detect(self, *args, **kwargs):
        return self._call("detect", *args, **kwargs)

    @property
    def name(self):
        self._ensure()
        return getattr(self._inst, "name", "")

    @property
    def lang(self):
        self._ensure()
        return getattr(self._inst, "lang", None)




def get_engine(name: str, cfg: Optional[Any] = None):
    """
    Factory: returns a LAZY engine proxy by name.
    The heavy engine is constructed only on first .ocr/.recognize call.
    """
    if not name:
        raise ValueError("get_engine: 'name' must be a non-empty string")

    key = str(name).strip().lower()
    if key not in _ENGINE_LOADERS:
        raise ValueError(
            f"Unknown OCR engine '{name}'. Valid options: {', '.join(ENGINE_ALIASES)}"
        )

    try:
        engine_cls = _ENGINE_LOADERS[key]()  # still just returns the class
    except ImportError as e:
        raise ImportError(
            f"Failed to import OCR engine '{name}'. "
            "Make sure its optional dependencies are installed."
        ) from e

    kwargs = _cfg_to_kwargs(cfg)
    # filter kwargs to ctor signature and inject cache_dir if supported
    try:
        import inspect
        sig = inspect.signature(engine_cls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        if "cache_dir" in allowed and "cache_dir" not in kwargs:
            try:
                from fait.core.paths import get_paths
                from fait.core.utils import ensure_folder
                cache_root = get_paths().models_cache / "ocr"
                ensure_folder(cache_root)
                kwargs["cache_dir"] = str(cache_root)
            except Exception:
                pass
    except Exception:
        pass

    # RETURN A LAZY PROXY, NOT THE REAL ENGINE
    return _LazyEngine(engine_cls, kwargs)
