# tests/conftest.py
import os, sys, pathlib
# make src importable for tests
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import types, pytest
from fait.core import paths as paths_mod  # now safe

@pytest.fixture
def tmp_paths(tmp_path, monkeypatch):
    p = types.SimpleNamespace(
        repo_root=tmp_path,
        fait_root=tmp_path / ".fait",
        cache_root=tmp_path / ".fait" / "cache",
        models_cache=tmp_path / ".fait" / "cache" / "models",
        models_facial_recognition=tmp_path / ".fait" / "cache" / "models" / "vision" / "face_match",
        models_object_detection=tmp_path / ".fait" / "cache" / "models" / "vision" / "object_screen",
        models_ocr=tmp_path / ".fait" / "cache" / "models" / "vision"/"ocr",
        embeddings_cache=tmp_path / ".fait" / "cache" / "models",
        outputs=tmp_path / ".fait" / "outputs",
        logs=tmp_path / ".fait" / "logs",
    )
    monkeypatch.setattr(paths_mod, "get_paths", lambda: p)
    return p

def _install_torch_stub():
    import types, sys
    torch = types.ModuleType("torch")

    class _Cuda:
        @staticmethod
        def is_available(): return False
    torch.cuda = _Cuda()

    def _noop_ctx(*a, **k):
        class _C:
            def __enter__(self): return None
            def __exit__(self, *e): return False
        return _C()
    def _decorator(*a, **k):
        def _wrap(fn): return fn
        return _wrap
    torch.no_grad = _noop_ctx
    torch.inference_mode = _decorator

    class _Tensor:
        def to(self, *a, **k): return self
        def eval(self): return self
    torch.Tensor = _Tensor

    # NEW: no-op persistence funcs
    def _save(*a, **k): return None
    def _load(*a, **k): return None
    torch.save = _save
    torch.load = _load

    torch.__version__ = "0.0.stub"
    sys.modules["torch"] = torch

def _install_paddleocr_stub():
    mod = types.ModuleType("paddleocr")
    class PaddleOCR:
        def __init__(self, *a, **k): pass
        def ocr(self, *a, **k): return [[{"dt_polys": []}]]
    mod.PaddleOCR = PaddleOCR
    sys.modules["paddleocr"] = mod

def _install_doctr_stub():
    root = types.ModuleType("doctr")
    root.__version__ = "stub"
    models = types.ModuleType("doctr.models")
    io = types.ModuleType("doctr.io")
    # predictor returns an object with .export()
    def ocr_predictor(*a, **k):
        class _P:
            def __call__(self, doc):
                class _Out:
                    def export(self): return {"pages": []}
                return _Out()
        return _P()
    models.ocr_predictor = ocr_predictor
    # very forgiving DocumentFile
    class DocumentFile:
        @classmethod
        def from_images(cls, images):
            return images
    io.DocumentFile = DocumentFile
    sys.modules["doctr"] = root
    sys.modules["doctr.models"] = models
    sys.modules["doctr.io"] = io

# Activate only on CI
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    # Keep native libs from fighting over threads even if something slips through
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    # Install stubs
    _install_torch_stub()
    _install_paddleocr_stub()
    _install_doctr_stub()

    import types

    ultra = types.ModuleType("ultralytics")


    class YOLO:
        def __init__(self, *a, **k): pass

        # match common calls in code/tests
        def predict(self, *a, **k): return []

        def __call__(self, *a, **k): return []


    ultra.YOLO = YOLO
    sys.modules["ultralytics"] = ultra
