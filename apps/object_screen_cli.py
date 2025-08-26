import argparse, pathlib, sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from dotenv import load_dotenv
from fait.core.logging_config import setup_logging
from fait.vision.pipelines.object_screen import ScreenConfig, run_object_screen

def main():
    load_dotenv()
    setup_logging()

    p = argparse.ArgumentParser("Open-vocab → Closed-set object screening")
    p.add_argument("--gallery-dir", required=True)
    p.add_argument("--gdino-only", action="store_true", help="Use GroundingDINO only (no closed-set verifier)")
    p.add_argument("--prompts", nargs="+", required=True, help='e.g. "handgun" "knife" "laptop"')
    p.add_argument("--output-dir", default=None)
    p.add_argument("--save-crops", action="store_true")
    p.add_argument("--and-rule", action="store_true", help="Use AND rule; default is weighted fusion")
    p.add_argument("--tau", type=float, default=0.60, help="Weighted fusion acceptance (τ*)")
    p.add_argument("--alpha", type=float, default=0.5, help="Fusion weight α for GDINO")
    p.add_argument("--gdino-box-thr", type=float, default=0.25)
    p.add_argument("--gdino-text-thr", type=float, default=0.25)
    args = p.parse_args()

    fcfg = dict(rule="and" if args.and_rule else "weighted", tau_star=args.tau, alpha=args.alpha)
    scfg = ScreenConfig(
        prompts=args.prompts,
        gallery_dir=args.gallery_dir,
        output_dir=args.output_dir,
        save_crops=args.save_crops,
        verifier="none" if args.gdino_only else "deformable_detr",
    )
    scfg.gdino.box_threshold = args.gdino_box_thr
    scfg.gdino.text_threshold = args.gdino_text_thr
    # tweak whitelist here if needed: scfg.detr.class_whitelist = [...]

    res = run_object_screen(scfg)
    print("Summary:", res)

if __name__ == "__main__":
    main()
