from dotenv import load_dotenv
load_dotenv()

import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import typer
from fait.core.registry import get_embedder
from fait.vision.pipelines.face_match import run_face_match
from fait.core.logging_config import setup_logging
setup_logging()


app = typer.Typer(help="FAIT — face recognition CLI")

@app.command()
def match(
    model: str = typer.Argument("arcface", help="arcface | clip"),
    reference_dir: str = typer.Option(..., "--reference-dir"),
    gallery_dir: str = typer.Option(..., "--gallery-dir"),
    output_dir: str | None = typer.Option(None, "--output-dir"),
    thresholds: str = typer.Option(..., help="Comma-separated thresholds"),
    metric: str = typer.Option("euclidean", help="euclidean (ArcFace) | cosine (CLIP via 1-cos)"),
    use_cache: bool = True,
    plot_results: bool = False,
    topk_preview: int = 10,
):
    ths = [float(x) for x in thresholds.split(",")]
    emb = get_embedder(model)
    res = run_face_match(
        embedder=emb,
        reference_dir=reference_dir,
        gallery_dir=gallery_dir,
        output_dir=output_dir,  # may be None → pipeline resolves
        thresholds=ths,
        metric=metric,
        use_cache=use_cache,
        plot_results=plot_results,
        topk_preview=topk_preview,
    )
    typer.echo(res["report_path"])

if __name__ == "__main__":
    app()


'''
Example usage:
python facial_recognition_arcface.py \
    --reference-dir ./path/to/reference_images --gallery-dir ./path/to/suspect_gallery --output-dir ./path/to/output_dir  \
    --thresholds 0.8 0.9 1.0 --distance-metric euclidean --plot-results --use-cache
'''