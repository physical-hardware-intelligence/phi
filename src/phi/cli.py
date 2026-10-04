"""Φ command-line entrypoint (thin).

Most real work is done by the LeRobot CLI (lerobot-record / -train / -rollout ...);
this wrapper exists to point students at the right documented command and, later,
to bundle Φ's dataset-QA / eval-report / edge-deploy helpers.
"""
from __future__ import annotations

from pathlib import Path

import typer

STUDIO_DATA = Path.home() / ".cache" / "phi" / "studio"
app = typer.Typer(help="Φ — Physical Hardware Intelligence (robot-learning pipeline).")


@app.command()
def where(stage: str) -> None:
    """Print the doc page for a pipeline stage: setup|data|train|eval|deploy."""
    pages = {
        "setup": "docs/robots/so-arm101/02-setup.md",
        "data": "docs/robots/so-arm101/03-teleop-and-data.md",
        "train": "docs/training/README.md",
        "eval": "docs/evaluation/README.md",
        "deploy": "docs/deployment/README.md",
    }
    typer.echo(pages.get(stage, "unknown stage — try: setup|data|train|eval|deploy"))


@app.command()
def studio(
    mock: bool = typer.Option(True, "--mock/--hardware", help="Mock rig, or real arms."),
    pairs: int = typer.Option(1, min=1, max=2, help="Leader/follower pairs: 1, or 2 for bimanual."),
    port: int = typer.Option(8765, help="Local port. Studio binds 127.0.0.1 only."),
    browser: bool = typer.Option(True, help="Open the browser."),
    data_dir: str = typer.Option(
        str(STUDIO_DATA), help="Where Studio keeps eval records and mock calibrations."
    ),
    assistant_model: str = typer.Option(
        "", help="Model for the Claude assistant, such as sonnet or opus. Empty: Claude's default."
    ),
) -> None:
    """Phi Studio: set up, calibrate, teleoperate, run and evaluate policies in a local app."""
    if not mock:
        typer.echo("The hardware backend is not built yet. Run with --mock for now.", err=True)
        raise typer.Exit(1)
    from phi.studio.server import PortInUse, serve

    try:
        serve({"kind": "mock", "pairs": pairs}, port=port, open_browser=browser,
              data_dir=Path(data_dir).expanduser(), assistant_model=assistant_model or None)
    except PortInUse as e:
        typer.echo(f"{e} Close it, or run with --port {e.port + 1}.", err=True)
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
