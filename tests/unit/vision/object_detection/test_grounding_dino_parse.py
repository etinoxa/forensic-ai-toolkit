# tests/unit/vision/test_grounding_dino_parse.py
import pytest
pytestmark = pytest.mark.requires_models
from PIL import Image
import fait.core.paths as paths_mod
import fait.vision.object_detection.models.grounding_dino as gdimod

def test_gdino_propose_shape(monkeypatch, tmp_paths):
    monkeypatch.setattr(paths_mod, "get_paths", lambda: tmp_paths, raising=False)

    class StubGD:
        def __init__(self, *_a, **_k): pass
        def propose(self, img_or_path, prompts):
            assert isinstance(prompts, list)
            return [{"box":[10,10,50,50], "score":0.61, "label":"gun"}]

    # If your code normally instantiates GroundingDINO(),
    # patch the class to our stub for this test
    monkeypatch.setattr(gdimod, "GroundingDINO", StubGD)

    gd = gdimod.GroundingDINO()
    props = gd.propose(Image.new("RGB",(16,16)), ["gun"])
    assert isinstance(props, list) and "box" in props[0] and "score" in props[0] and "label" in props[0]
