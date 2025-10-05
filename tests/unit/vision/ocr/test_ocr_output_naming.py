# tests/unit/vision/test_ocr_output_naming.py
from PIL import Image
import fait.vision.pipelines.ocr_pipeline as ocrmod


def _mk_cfg(tmp_paths, gal, strategy, verifier):
    class Cfg: pass

    cfg = Cfg()
    cfg.strategy = strategy
    cfg.verifier = verifier
    cfg.gallery_dir = str(gal)
    cfg.output_dir = str(tmp_paths.outputs)
    cfg.rotations = [0]
    cfg.exts = [".png"]
    cfg.min_file_kb = 0
    cfg.min_dim_px = [0, 0]
    # models presence just to satisfy pipeline guards
    cfg.engines = {"paddle": {"enabled": True}, "trocr": {"enabled": True}, "tesseract": {"enabled": True}}
    cfg.fusion = ocrmod.FusionCfg()
    cfg.engine_order = ["paddle", "trocr", "tesseract"]

    return cfg  # Return the config object, not the run_ocr result


def test_output_naming_contains_strategy(monkeypatch, tmp_paths, tmp_path):
    monkeypatch.setattr(ocrmod, "get_paths", lambda: tmp_paths, raising=False)
    gal = tmp_path / "gal";
    gal.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 16), "white").save(gal / "a.png")

    # Detector only
    cfg = _mk_cfg(tmp_paths, gal, "detector_only", "tesseract")
    out = ocrmod.run_ocr(cfg)

    # Debug: Check what's actually returned
    print(f"run_ocr returned: {out}")
    print(f"Type: {type(out)}")
    if isinstance(out, dict):
        print(f"Keys: {list(out.keys())}")

    # Check if run_dir exists or use alternative approach
    if isinstance(out, dict) and "run_dir" in out:
        assert "detector_only" in out["run_dir"]
    else:
        # Alternative: check output directory directly
        output_files = list(tmp_paths.outputs.glob("*detector_only*"))
        assert len(output_files) > 0

