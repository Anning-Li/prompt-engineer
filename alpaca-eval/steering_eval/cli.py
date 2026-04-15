from __future__ import annotations

import logging
from pathlib import Path

import typer

from . import config
from .alpaca_dataset import build_manifest
from .alpaca_scoring import score_alpaca
from .generation import run_alpaca_generation, validate_generation_matrix
from .reporting import report_alpaca

app = typer.Typer(help="Alpaca-first steering experiment harness.")


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@app.command("prep-alpaca")
def prep_alpaca(
    sample_size: int = typer.Option(config.DEFAULT_SAMPLE_SIZE, min=1),
    seed: int = typer.Option(config.DEFAULT_SEED),
    force: bool = typer.Option(False, help="Rebuild the manifest even if it already exists."),
) -> None:
    _setup_logging()
    manifest = build_manifest(sample_size=sample_size, seed=seed, force=force)
    typer.echo(f"Wrote {len(manifest)} Alpaca prompts to {config.manifest_path(sample_size, seed)}")


@app.command("run-alpaca")
def run_alpaca(
    generator_model: str = typer.Option(..., help="Model used to generate Alpaca responses."),
    manifest: Path | None = typer.Option(None, dir_okay=False),
    limit_prompts: int | None = typer.Option(None, min=1),
    max_output_tokens: int = typer.Option(config.DEFAULT_MAX_OUTPUT_TOKENS, min=1),
    force: bool = typer.Option(False, help="Regenerate outputs even if cached records already exist."),
) -> None:
    _setup_logging()
    records = run_alpaca_generation(
        generator_model=generator_model,
        manifest_path=manifest,
        limit_prompts=limit_prompts,
        max_output_tokens=max_output_tokens,
        force=force,
    )
    typer.echo(f"Generated or loaded {len(records)} final Alpaca answers for {generator_model}.")


@app.command("validate-alpaca")
def validate_alpaca(
    generator_model: str = typer.Option(..., help="Model used to generate Alpaca responses."),
    manifest: Path | None = typer.Option(None, dir_okay=False),
    limit_prompts: int | None = typer.Option(None, min=1),
) -> None:
    _setup_logging()
    missing = validate_generation_matrix(
        generator_model=generator_model,
        manifest_path=manifest,
        limit_prompts=limit_prompts,
    )
    if missing:
        typer.echo(f"Generation matrix incomplete: {len(missing)} missing cells.")
        for sample in missing[:20]:
            typer.echo(f"  {sample}")
        raise typer.Exit(code=1)
    typer.echo("Generation matrix is complete.")


@app.command("score-alpaca")
def score_alpaca_command(
    generator_model: str = typer.Option(..., help="Model used to generate Alpaca responses."),
    manifest: Path | None = typer.Option(None, dir_okay=False),
    limit_prompts: int | None = typer.Option(None, min=1),
    judge_model: str = typer.Option(config.DEFAULT_CORRECTNESS_JUDGE_MODEL, help="Judge model for correctness."),
    run_id: str | None = typer.Option(None),
    force: bool = typer.Option(False, help="Recompute judge scores even if cached judgments already exist."),
) -> None:
    _setup_logging()
    df = score_alpaca(
        generator_model=generator_model,
        manifest_path=manifest,
        limit_prompts=limit_prompts,
        judge_model=judge_model,
        run_id=run_id,
        force=force,
    )
    typer.echo(f"Wrote {len(df)} Alpaca score rows to {config.SCORES_DIR / 'alpaca_eval_summary.csv'}")


@app.command("report-alpaca")
def report_alpaca_command(
    run_id: str | None = typer.Option(None),
) -> None:
    _setup_logging()
    outputs = report_alpaca(run_id=run_id)
    for key, path in outputs.items():
        typer.echo(f"{key}: {path}")


if __name__ == "__main__":
    app()
