# apps/face_match_cli.py
import typer
from fod.embeddings.arcface import ArcFaceEmbedder
from fod.pipelines.face_matcher import FaceMatcher

app = typer.Typer()

@app.command()
def match(
    reference_dir: str,
    gallery_dir: str,
    output_dir: str,
    thresholds: str,
    metric: str = typer.Option("euclidean", help="euclidean|cosine"),
    use_cache: bool = True,
    plot_results: bool = False,
    topk_preview: int = 10,
):
    ths = [float(x) for x in thresholds.split(",")]
    matcher = FaceMatcher(ArcFaceEmbedder(), metric=metric)
    res = matcher.run(reference_dir, gallery_dir, output_dir, ths, use_cache, plot_results, topk_preview)
    typer.echo(res)

if __name__ == "__main__":
    app()
