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
        models_ocr=tmp_path / ".fait" / "cache" / "models" / "vision" / "ocr",
        embeddings_cache=tmp_path / ".fait" / "cache" / "models",
        outputs=tmp_path / ".fait" / "outputs",
        logs=tmp_path / ".fait" / "logs",
    )
    monkeypatch.setattr(paths_mod, "get_paths", lambda: p)
    return p


def _install_torch_stub():
    import types, sys
    from importlib.machinery import ModuleSpec

    torch = types.ModuleType("torch")
    torch.__spec__ = ModuleSpec("torch", None)
    torch.__file__ = "<stub>"
    torch.__package__ = "torch"

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

    # No-op persistence funcs
    def _save(*a, **k): return None

    def _load(*a, **k): return None

    torch.save = _save
    torch.load = _load

    # Add dtype stubs for safetensors
    class _DType:
        def __init__(self, name): self.name = name

        def __repr__(self): return f"torch.{self.name}"

    torch.dtype = _DType  # The type itself
    torch.float32 = _DType("float32")
    torch.float16 = _DType("float16")
    torch.bfloat16 = _DType("bfloat16")
    torch.float64 = _DType("float64")
    torch.int64 = _DType("int64")
    torch.int32 = _DType("int32")
    torch.int16 = _DType("int16")
    torch.int8 = _DType("int8")
    torch.uint8 = _DType("uint8")
    torch.bool = _DType("bool")

    # Add hub module for torchvision compatibility
    hub = types.ModuleType("torch.hub")
    hub.__spec__ = ModuleSpec("torch.hub", None)

    def _get_torch_home():
        """Stub for torch.hub.get_torch_home()"""
        return os.path.expanduser("~/.cache/torch")

    hub.get_torch_home = _get_torch_home
    torch.hub = hub
    sys.modules["torch.hub"] = hub

    # Add nn module for safetensors/transformers/speechbrain compatibility
    nn = types.ModuleType("torch.nn")
    nn.__spec__ = ModuleSpec("torch.nn", None)

    class Module:
        """Stub for torch.nn.Module"""

        def __init__(self): pass

        def to(self, *a, **k): return self

        def eval(self): return self

        def train(self, mode=True): return self

        def parameters(self): return []

        def named_parameters(self): return []

        def state_dict(self): return {}

        def load_state_dict(self, state_dict): pass

    # DataParallel and DistributedDataParallel
    class DataParallel(Module):
        """Stub for DP"""

        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module

    class DistributedDataParallel(Module):
        """Stub for DDP"""

        def __init__(self, module, *args, **kwargs):
            super().__init__()
            self.module = module

    nn.Module = Module
    nn.DataParallel = DataParallel
    nn.DistributedDataParallel = DistributedDataParallel

    # Add functional module
    functional = types.ModuleType("torch.nn.functional")
    functional.__spec__ = ModuleSpec("torch.nn.functional", None)

    def normalize(input, p=2, dim=1, eps=1e-12):
        """Stub for F.normalize"""
        return input

    functional.normalize = normalize
    nn.functional = functional
    sys.modules["torch.nn.functional"] = functional

    # Add parallel module for SpeechBrain/DDP
    parallel = types.ModuleType("torch.nn.parallel")
    parallel.__spec__ = ModuleSpec("torch.nn.parallel", None)
    parallel.DataParallel = DataParallel
    parallel.DistributedDataParallel = DistributedDataParallel
    nn.parallel = parallel
    sys.modules["torch.nn.parallel"] = parallel

    torch.nn = nn
    sys.modules["torch.nn"] = nn

    # Add distributed module stub
    distributed = types.ModuleType("torch.distributed")
    distributed.__spec__ = ModuleSpec("torch.distributed", None)
    distributed.is_available = lambda: False
    distributed.is_initialized = lambda: False
    distributed.get_rank = lambda: 0
    distributed.get_world_size = lambda: 1
    torch.distributed = distributed
    sys.modules["torch.distributed"] = distributed

    # Add utils module and submodules
    utils = types.ModuleType("torch.utils")
    utils.__spec__ = ModuleSpec("torch.utils", None)
    torch.utils = utils
    sys.modules["torch.utils"] = utils

    _pytree = types.ModuleType("torch.utils._pytree")
    _pytree.__spec__ = ModuleSpec("torch.utils._pytree", None)

    # Add no-op register function
    def register_pytree_node(*args, **kwargs):
        pass

    _pytree.register_pytree_node = register_pytree_node
    utils._pytree = _pytree
    sys.modules["torch.utils._pytree"] = _pytree

    # Add data module for DataLoader compatibility
    data = types.ModuleType("torch.utils.data")
    data.__spec__ = ModuleSpec("torch.utils.data", None)

    class DataLoader:
        def __init__(self, *args, **kwargs): pass

        def __iter__(self): return iter([])

    data.DataLoader = DataLoader
    utils.data = data
    sys.modules["torch.utils.data"] = data

    # Add tensor creation functions
    def tensor(*args, **kwargs):
        """Stub tensor creation"""
        return _Tensor()

    def zeros(*args, **kwargs):
        return _Tensor()

    def ones(*args, **kwargs):
        return _Tensor()

    torch.tensor = tensor
    torch.zeros = zeros
    torch.ones = ones
    torch.as_tensor = tensor

    torch.__version__ = "2.0.0"
    sys.modules["torch"] = torch


def _install_paddleocr_stub():
    import types
    from importlib.machinery import ModuleSpec

    mod = types.ModuleType("paddleocr")
    mod.__spec__ = ModuleSpec("paddleocr", None)

    class PaddleOCR:
        def __init__(self, *a, **k): pass

        def ocr(self, *a, **k): return [[{"dt_polys": []}]]

    mod.PaddleOCR = PaddleOCR
    sys.modules["paddleocr"] = mod

def _install_doctr_stub():
    import types
    from importlib.machinery import ModuleSpec

    root = types.ModuleType("doctr")
    root.__version__ = "stub"
    root.__spec__ = ModuleSpec("doctr", None)

    models = types.ModuleType("doctr.models")
    models.__spec__ = ModuleSpec("doctr.models", None)

    io = types.ModuleType("doctr.io")
    io.__spec__ = ModuleSpec("doctr.io", None)

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

def _install_torchvision_stub():
    """Stub torchvision to prevent import errors"""
    import types, sys
    from importlib.machinery import ModuleSpec

    # Main torchvision module
    tv = types.ModuleType("torchvision")
    tv.__version__ = "0.15.0"
    tv.__spec__ = ModuleSpec("torchvision", None)
    tv.__file__ = "<stub>"
    tv.__package__ = "torchvision"

    # torchvision.ops module (needed by grounding_dino)
    ops = types.ModuleType("torchvision.ops")
    ops.__spec__ = ModuleSpec("torchvision.ops", None)
    ops.__package__ = "torchvision.ops"

    # Add nms function stub
    def nms(boxes, scores, iou_threshold):
        """Stub NMS that returns empty indices"""
        import torch
        return torch.tensor([], dtype=torch.long)

    ops.nms = nms
    tv.ops = ops

    # torchvision.transforms module (needed by transformers)
    transforms = types.ModuleType("torchvision.transforms")
    transforms.__spec__ = ModuleSpec("torchvision.transforms", None)
    transforms.__package__ = "torchvision.transforms"

    # Add InterpolationMode enum stub
    class InterpolationMode:
        NEAREST = 0
        BILINEAR = 2
        BICUBIC = 3

    transforms.InterpolationMode = InterpolationMode
    tv.transforms = transforms

    # Extension stubs to prevent initialization errors
    extension = types.ModuleType("torchvision.extension")
    extension.__spec__ = ModuleSpec("torchvision.extension", None)
    extension._HAS_OPS = False
    tv.extension = extension

    _internally_replaced_utils = types.ModuleType("torchvision._internally_replaced_utils")
    _internally_replaced_utils.__spec__ = ModuleSpec("torchvision._internally_replaced_utils", None)

    def _get_extension_path(lib_name):
        return ""

    _internally_replaced_utils._get_extension_path = _get_extension_path
    tv._internally_replaced_utils = _internally_replaced_utils

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.extension"] = extension
    sys.modules["torchvision._internally_replaced_utils"] = _internally_replaced_utils


def _install_speechbrain_stub():
    """Stub SpeechBrain to avoid torch dependency chains"""
    import types, sys
    from importlib.machinery import ModuleSpec

    # Main speechbrain module
    sb = types.ModuleType("speechbrain")
    sb.__spec__ = ModuleSpec("speechbrain", None)
    sb.__version__ = "1.0.0"

    # speechbrain.utils
    utils = types.ModuleType("speechbrain.utils")
    utils.__spec__ = ModuleSpec("speechbrain.utils", None)
    sb.utils = utils

    # speechbrain.utils.fetching
    fetching = types.ModuleType("speechbrain.utils.fetching")
    fetching.__spec__ = ModuleSpec("speechbrain.utils.fetching", None)

    def fetch(*args, **kwargs):
        """Stub fetch function"""
        return "/tmp/stub"

    fetching.fetch = fetch
    utils.fetching = fetching

    # speechbrain.inference
    inference = types.ModuleType("speechbrain.inference")
    inference.__spec__ = ModuleSpec("speechbrain.inference", None)
    sb.inference = inference

    # speechbrain.inference.speaker
    speaker = types.ModuleType("speechbrain.inference.speaker")
    speaker.__spec__ = ModuleSpec("speechbrain.inference.speaker", None)
    inference.speaker = speaker

    # EncoderClassifier - needs to be in both places
    class EncoderClassifier:
        """Stub for SpeechBrain EncoderClassifier"""

        def __init__(self, *args, **kwargs):
            pass

        @classmethod
        def from_hparams(cls, *args, **kwargs):
            return cls()

        def eval(self):
            return self

        def encode_batch(self, wav):
            """Return fake embedding"""
            import numpy as np
            # Return a fake tensor-like object
            class FakeTensor:
                def squeeze(self, *args):
                    return self

                def cpu(self):
                    return self

                def numpy(self):
                    return np.zeros(192, dtype=np.float32)

            return FakeTensor()

    # Make EncoderClassifier available from both locations
    inference.EncoderClassifier = EncoderClassifier
    speaker.EncoderClassifier = EncoderClassifier

    # Register all modules
    sys.modules["speechbrain"] = sb
    sys.modules["speechbrain.utils"] = utils
    sys.modules["speechbrain.utils.fetching"] = fetching
    sys.modules["speechbrain.inference"] = inference
    sys.modules["speechbrain.inference.speaker"] = speaker
# Activate only on CI
if os.getenv("GITHUB_ACTIONS") == "true" or os.getenv("CI") == "true":
    # Keep native libs from fighting over threads even if something slips through
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

    # Install stubs in order
    _install_torch_stub()
    _install_torchvision_stub()  # Must come after torch stub
    _install_paddleocr_stub()
    _install_doctr_stub()
    _install_speechbrain_stub()

    # Stub transformers to prevent heavy imports
    import types
    from importlib.machinery import ModuleSpec

    transformers = types.ModuleType("transformers")
    transformers.__spec__ = ModuleSpec("transformers", None)
    transformers.__version__ = "4.43.4"

    # Add commonly used transformers classes as mocks
    from unittest.mock import MagicMock

    transformers.AutoImageProcessor = MagicMock()
    transformers.AutoModelForObjectDetection = MagicMock()
    transformers.AutoProcessor = MagicMock()
    transformers.AutoModelForZeroShotObjectDetection = MagicMock()
    transformers.TrOCRProcessor = MagicMock()
    transformers.VisionEncoderDecoderModel = MagicMock()
    transformers.DonutProcessor = MagicMock()
    transformers.CLIPProcessor = MagicMock()
    transformers.CLIPModel = MagicMock()
    transformers.AutoFeatureExtractor = MagicMock()
    transformers.WavLMForXVector = MagicMock()

    sys.modules["transformers"] = transformers

    # Ultralytics stub
    ultra = types.ModuleType("ultralytics")
    ultra.__spec__ = ModuleSpec("ultralytics", None)


    class YOLO:
        def __init__(self, *a, **k): pass

        # match common calls in code/tests
        def predict(self, *a, **k): return []

        def __call__(self, *a, **k): return []


    ultra.YOLO = YOLO
    sys.modules["ultralytics"] = ultra