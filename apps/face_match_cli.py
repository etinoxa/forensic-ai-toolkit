# apps/face_match_cli.py
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import typer
from fait.core.registry import get_embedder
from fait.vision.pipelines.face_match import run_face_match

app = typer.Typer(help="FAIT — face recognition CLI")

@app.command()
def match(
    model: str = typer.Argument("arcface", help="arcface | clip"),
    reference_dir: str = typer.Option(..., "--reference-dir"),
    gallery_dir: str = typer.Option(..., "--gallery-dir"),
    output_dir: str = typer.Option(..., "--output-dir"),
    thresholds: str = typer.Option(..., help="Comma-separated thresholds"),
    metric: str = typer.Option("euclidean", help="euclidean (ArcFace) | cosine (CLIP via 1-cos)"),
    use_cache: bool = True,
    plot_results: bool = False,
    topk_preview: int = 10,
):
    ths = [float(x) for x in thresholds.split(",")]
    emb = get_embedder(model)
    res = run_face_match(emb, reference_dir, gallery_dir, output_dir, ths, metric, use_cache, plot_results, topk_preview)
    typer.echo(res)

if __name__ == "__main__":
    app()
