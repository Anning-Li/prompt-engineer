from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ModuleNotFoundError:  # pragma: no cover
    sns = None

from . import config
from .conditions import CONDITION_ORDER, CONSTRAINT_ORDER, PERSONA_ORDER, REFLECTION_ORDER

if sns is not None:
    sns.set_theme(style="whitegrid")


def _latest_file(path: Path, pattern: str) -> Path | None:
    candidates = sorted(path.glob(pattern))
    return candidates[-1] if candidates else None


def load_summary_table(run_id: str | None = None) -> pd.DataFrame:
    if run_id:
        summary_path = config.SCORES_DIR / f"alpaca_eval_summary_{run_id}.csv"
    else:
        summary_path = config.SCORES_DIR / "alpaca_eval_summary.csv"
    if not summary_path.exists():
        latest = _latest_file(config.SCORES_DIR, "alpaca_eval_summary_*.csv")
        if latest is None:
            raise FileNotFoundError("No Alpaca summary table found.")
        summary_path = latest
    return pd.read_csv(summary_path)


def build_pivot_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    ordered = summary_df.copy()
    ordered["condition_id"] = pd.Categorical(ordered["condition_id"], categories=CONDITION_ORDER, ordered=True)
    ordered = ordered.sort_values("condition_id")

    pivot = ordered.pivot_table(
        index=["persona_id", "constraint_id"],
        columns="reflection_mode",
        values="mean_correctness_score",
        aggfunc="first",
    ).reset_index()
    pivot = pivot.rename_axis(columns=None)

    row_order = {
        (persona_id, constraint_id): index
        for index, (persona_id, constraint_id) in enumerate(
            (persona_id, constraint_id)
            for persona_id in PERSONA_ORDER
            for constraint_id in CONSTRAINT_ORDER
        )
    }
    pivot["_row_order"] = pivot.apply(lambda row: row_order[(row["persona_id"], row["constraint_id"])], axis=1)
    pivot = pivot.sort_values("_row_order").drop(columns="_row_order")
    return pivot[["persona_id", "constraint_id", *REFLECTION_ORDER]]


def _display_pivot_table(pivot_df: pd.DataFrame) -> pd.DataFrame:
    display_df = pivot_df.copy()
    display_df["persona"] = display_df["persona_id"].map({"A": "Persona A", "B": "Persona B", "C": "Persona C"})
    display_df["constraint"] = display_df["constraint_id"].map(
        {"C1": "Constraint 1", "C2": "Constraint 2", "C3": "Constraint 3"}
    )
    for reflection_mode in REFLECTION_ORDER:
        display_df[reflection_mode] = display_df[reflection_mode].map(lambda value: f"{value:.2f}")
    return display_df[["persona", "constraint", *REFLECTION_ORDER]]


def write_pivot_artifacts(summary_df: pd.DataFrame) -> dict[str, Path]:
    pivot_df = build_pivot_table(summary_df)
    pivot_csv = config.SCORES_DIR / "alpaca_eval_pivot.csv"
    pivot_parquet = config.SCORES_DIR / "alpaca_eval_pivot.parquet"
    pivot_markdown = config.REPORTS_DIR / "alpaca_eval_matrix.md"
    pivot_df.to_csv(pivot_csv, index=False)
    pivot_df.to_parquet(pivot_parquet, index=False)
    display_df = _display_pivot_table(pivot_df)
    pivot_markdown.write_text(_markdown_table(display_df), encoding="utf-8")
    return {
        "pivot_csv": pivot_csv,
        "pivot_parquet": pivot_parquet,
        "pivot_markdown": pivot_markdown,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in headers) + " |")
    return "\n".join(lines) + "\n"


def plot_correctness_heatmap(summary_df: pd.DataFrame, output_path: Path) -> None:
    pivot_df = build_pivot_table(summary_df)
    heatmap_source = pivot_df.set_index(["persona_id", "constraint_id"])[REFLECTION_ORDER]
    y_labels = [f"{persona_id} / {constraint_id}" for persona_id, constraint_id in heatmap_source.index]

    fig, axis = plt.subplots(figsize=(7, 5))
    if sns is not None:
        sns.heatmap(
            heatmap_source,
            ax=axis,
            annot=True,
            fmt=".2f",
            cmap="crest",
            vmin=1,
            vmax=5,
            cbar=True,
        )
    else:
        image = axis.imshow(heatmap_source.values, vmin=1, vmax=5, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(REFLECTION_ORDER)))
        axis.set_xticklabels(REFLECTION_ORDER)
        axis.set_yticks(range(len(y_labels)))
        axis.set_yticklabels(y_labels)
        for row_index in range(heatmap_source.shape[0]):
            for column_index in range(heatmap_source.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{heatmap_source.iloc[row_index, column_index]:.2f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis)
    axis.set_xlabel("Reflection Mode")
    axis.set_ylabel("Persona / Constraint")
    axis.set_title("Alpaca Mean Correctness Score")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _reflection_summary(summary_df: pd.DataFrame) -> list[str]:
    lines = ["## Reflection Trends", ""]
    mean_by_mode = (
        summary_df.groupby("reflection_mode", as_index=False)["mean_correctness_score"]
        .mean()
        .sort_values("mean_correctness_score", ascending=False)
    )
    best = mean_by_mode.iloc[0]
    worst = mean_by_mode.iloc[-1]
    lines.append(
        f"- Highest overall mean correctness was `{best['reflection_mode']}` at {best['mean_correctness_score']:.2f}."
    )
    lines.append(
        f"- Lowest overall mean correctness was `{worst['reflection_mode']}` at {worst['mean_correctness_score']:.2f}."
    )
    lines.append("")
    return lines


def _row_summaries(summary_df: pd.DataFrame) -> list[str]:
    lines = ["## Row Summaries", ""]
    for persona_id in PERSONA_ORDER:
        for constraint_id in CONSTRAINT_ORDER:
            row_df = summary_df[
                (summary_df["persona_id"] == persona_id) & (summary_df["constraint_id"] == constraint_id)
            ].sort_values("reflection_mode")
            best_row = row_df.sort_values("mean_correctness_score", ascending=False).iloc[0]
            worst_row = row_df.sort_values("mean_correctness_score", ascending=True).iloc[0]
            lines.append(
                f"- `{persona_id} / {constraint_id}`: best `{best_row['reflection_mode']}` "
                f"({best_row['mean_correctness_score']:.2f}), worst `{worst_row['reflection_mode']}` "
                f"({worst_row['mean_correctness_score']:.2f})."
            )
    lines.append("")
    return lines


def write_markdown_report(summary_df: pd.DataFrame, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generator_models = sorted(summary_df["generator_model"].drop_duplicates())
    lines = [
        "# Alpaca 36-Cell Steering Report",
        "",
        "## Overview",
        "",
        f"- Generator model(s): {', '.join(f'`{model}`' for model in generator_models)}",
        f"- Evaluated `{summary_df['condition_id'].nunique()}` condition cells across `{int(summary_df['n_prompts'].max())}` prompts.",
        f"- The main metric is mean LLM-judge correctness score on a 1-5 scale.",
        "",
    ]
    lines.extend(_reflection_summary(summary_df))
    lines.extend(_row_summaries(summary_df))
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def report_alpaca(run_id: str | None = None) -> dict[str, Path]:
    config.ensure_directories()
    summary_df = load_summary_table(run_id=run_id)
    pivot_outputs = write_pivot_artifacts(summary_df)
    heatmap_path = config.PLOTS_DIR / "alpaca_eval_correctness_heatmap.png"
    report_path = config.REPORTS_DIR / "alpaca_eval_report.md"

    plot_correctness_heatmap(summary_df, heatmap_path)
    write_markdown_report(summary_df, report_path)
    return {
        "summary_csv": config.SCORES_DIR / "alpaca_eval_summary.csv",
        "pivot_csv": pivot_outputs["pivot_csv"],
        "pivot_markdown": pivot_outputs["pivot_markdown"],
        "heatmap": heatmap_path,
        "report": report_path,
    }
