from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config
from .alpaca_dataset import read_manifest
from .conditions import CONDITION_ORDER, condition_lookup
from .generation import load_final_answer_records, validate_generation_matrix
from .models import AggregateMetricRecord, CorrectnessJudgmentRecord, FinalAnswerRecord
from .utils import extract_json_object, iter_json_files, read_json, sha256_text, timestamp_run_id, write_json

LOGGER = logging.getLogger(__name__)


JUDGE_PROMPT = """You are evaluating a model answer against an instruction and a reference answer.

Score the model answer on a 1-5 integer scale using these anchors:
1 = incorrect, fails the task, or clearly worse than the reference answer
2 = major correctness or task-completion problems
3 = partially correct or somewhat useful, but meaningfully incomplete or flawed
4 = mostly correct and useful, with only minor issues
5 = fully correct, helpful, and well-aligned with the instruction

Judge the model answer primarily for correctness and task completion. Do not penalize harmless wording differences from the reference answer. If the model is appropriately uncertain, that can still score well. Return only JSON with this schema:
{"score": int, "reason": str}
"""


def parse_correctness_judgment_response(text: str) -> dict[str, Any]:
    payload = extract_json_object(text)
    score = int(payload["score"])
    if score < 1 or score > 5:
        raise ValueError(f"Judge score must be in [1, 5], received {score}.")
    reason = str(payload["reason"]).strip()
    if not reason:
        raise ValueError("Judge response must include a non-empty reason.")
    return {"score": score, "reason": reason}


def judgment_path_for_key(cache_key: str) -> Path:
    return config.JUDGMENTS_DIR / f"{cache_key}.json"


def load_correctness_judgments() -> list[CorrectnessJudgmentRecord]:
    records: list[CorrectnessJudgmentRecord] = []
    for path in iter_json_files(config.JUDGMENTS_DIR):
        records.append(CorrectnessJudgmentRecord.model_validate(read_json(path)))
    return records


def _final_answers_for_model(
    manifest_path: Path | None,
    generator_model: str,
    limit_prompts: int | None = None,
) -> list[FinalAnswerRecord]:
    manifest_entries = read_manifest(manifest_path)
    if limit_prompts is not None:
        manifest_entries = manifest_entries[:limit_prompts]
    prompt_order = {entry.prompt_id: index for index, entry in enumerate(manifest_entries)}
    records = [
        record
        for record in load_final_answer_records()
        if record.generator_model == generator_model and record.prompt_id in prompt_order
    ]
    return sorted(records, key=lambda record: (prompt_order[record.prompt_id], CONDITION_ORDER.index(record.condition_id)))


def _judgment_cache_key(
    final_answer: FinalAnswerRecord,
    judge_model: str,
) -> str:
    return sha256_text(
        "::".join(
            [
                final_answer.dataset,
                final_answer.prompt_id,
                final_answer.condition_id,
                final_answer.generator_model,
                judge_model,
                sha256_text(final_answer.final_response_text),
                sha256_text(final_answer.reference_output),
            ]
        )
    )


def _judge_input_text(final_answer: FinalAnswerRecord) -> str:
    return "\n\n".join(
        [
            f"Instruction:\n{final_answer.prompt_text}",
            f"Reference Answer:\n{final_answer.reference_output}",
            f"Model Answer:\n{final_answer.final_response_text}",
        ]
    )


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _judge_response(client: OpenAI, judge_model: str, input_text: str) -> Any:
    return client.responses.create(
        model=judge_model,
        instructions=JUDGE_PROMPT,
        input=input_text,
        temperature=0,
        top_p=1,
        max_output_tokens=256,
    )


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                pieces.append(text_value)
    text = "\n".join(pieces).strip()
    if not text:
        raise ValueError("The judge response did not contain any text output.")
    return text


def score_alpaca(
    generator_model: str,
    manifest_path: Path | None = None,
    limit_prompts: int | None = None,
    judge_model: str = config.DEFAULT_CORRECTNESS_JUDGE_MODEL,
    run_id: str | None = None,
    force: bool = False,
) -> pd.DataFrame:
    config.ensure_directories()
    missing = validate_generation_matrix(
        generator_model=generator_model,
        manifest_path=manifest_path,
        limit_prompts=limit_prompts,
    )
    if missing:
        raise RuntimeError(
            "Cannot score Alpaca outputs before the generation matrix is complete. "
            f"Missing {len(missing)} cells, for example: {missing[:5]}"
        )

    manifest_entries = read_manifest(manifest_path)
    if limit_prompts is not None:
        manifest_entries = manifest_entries[:limit_prompts]
    expected_count = len(manifest_entries)
    final_answers = _final_answers_for_model(
        manifest_path=manifest_path,
        generator_model=generator_model,
        limit_prompts=limit_prompts,
    )
    if len(final_answers) != expected_count * len(CONDITION_ORDER):
        raise RuntimeError(
            f"Expected {expected_count * len(CONDITION_ORDER)} final answers for {generator_model}, "
            f"found {len(final_answers)}."
        )

    client = OpenAI()
    conditions = condition_lookup()
    judgment_records: list[CorrectnessJudgmentRecord] = []
    for final_answer in final_answers:
        cache_key = _judgment_cache_key(final_answer, judge_model=judge_model)
        cache_path = judgment_path_for_key(cache_key)
        if cache_path.exists() and not force:
            judgment = CorrectnessJudgmentRecord.model_validate(read_json(cache_path))
        else:
            response = _judge_response(
                client=client,
                judge_model=judge_model,
                input_text=_judge_input_text(final_answer),
            )
            parsed = parse_correctness_judgment_response(_extract_output_text(response))
            judgment = CorrectnessJudgmentRecord(
                prompt_id=final_answer.prompt_id,
                condition_id=final_answer.condition_id,
                generator_model=final_answer.generator_model,
                persona_id=final_answer.persona_id,
                constraint_id=final_answer.constraint_id,
                reflection_mode=final_answer.reflection_mode,
                score=parsed["score"],
                reason=parsed["reason"],
                judge_model=judge_model,
                cache_key=cache_key,
            )
            write_json(cache_path, judgment.model_dump())
        judgment_records.append(judgment)

    write_judgment_index(load_correctness_judgments())

    run_identifier = run_id or timestamp_run_id("alpaca-score")
    aggregate_rows: list[AggregateMetricRecord] = []
    judgment_df = pd.DataFrame([record.model_dump() for record in judgment_records])
    for condition_id in CONDITION_ORDER:
        condition = conditions[condition_id]
        condition_rows = judgment_df[judgment_df["condition_id"] == condition_id]
        if len(condition_rows) != expected_count:
            raise RuntimeError(
                f"Expected {expected_count} judgments for {generator_model}/{condition_id}, "
                f"found {len(condition_rows)}."
            )
        aggregate_rows.append(
            AggregateMetricRecord(
                generator_model=generator_model,
                persona_id=condition.persona_id,
                constraint_id=condition.constraint_id,
                reflection_mode=condition.reflection_mode,
                condition_id=condition_id,
                mean_correctness_score=float(condition_rows["score"].mean()),
                n_prompts=expected_count,
                run_id=run_identifier,
            )
        )

    summary_df = pd.DataFrame([row.model_dump() for row in aggregate_rows]).sort_values(
        ["persona_id", "constraint_id", "reflection_mode"]
    )
    summary_csv = config.SCORES_DIR / "alpaca_eval_summary.csv"
    summary_parquet = config.SCORES_DIR / "alpaca_eval_summary.parquet"
    run_csv = config.SCORES_DIR / f"alpaca_eval_summary_{run_identifier}.csv"
    run_parquet = config.SCORES_DIR / f"alpaca_eval_summary_{run_identifier}.parquet"
    summary_df.to_csv(summary_csv, index=False)
    summary_df.to_parquet(summary_parquet, index=False)
    summary_df.to_csv(run_csv, index=False)
    summary_df.to_parquet(run_parquet, index=False)
    LOGGER.info("Wrote Alpaca correctness summary to %s", summary_csv)
    return summary_df


def write_judgment_index(records: list[CorrectnessJudgmentRecord]) -> None:
    if not records:
        return
    df = pd.DataFrame([record.model_dump() for record in records]).sort_values(
        ["generator_model", "condition_id", "prompt_id"]
    )
    df.to_csv(config.ARTIFACTS_DIR / "alpaca_eval_correctness_judgments.csv", index=False)
    df.to_parquet(config.ARTIFACTS_DIR / "alpaca_eval_correctness_judgments.parquet", index=False)
