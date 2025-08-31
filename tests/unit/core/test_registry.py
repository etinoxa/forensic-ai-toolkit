import importlib
from fait.core import registry

def test_get_embedder_unknown():
    try:
        registry._EMBEDDERS.clear()
        assert list(registry._EMBEDDERS) == []
        try:
            registry.get_embedder("nope")
        except KeyError:
            pass
        else:
            assert False, "expected KeyError"
    finally:
        importlib.reload(registry)
