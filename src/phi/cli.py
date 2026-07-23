"""Φ command-line entrypoint (thin).

Most real work is done by the LeRobot CLI (lerobot-record / -train / -rollout ...);
this wrapper exists to point students at the right documented command and, later,
to bundle Φ's dataset-QA / eval-report / edge-deploy helpers.
"""
from __future__ import annotations

import typer

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


if __name__ == "__main__":
    app()
