from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from . import config
from .alpaca_dataset import read_manifest
from .conditions import build_conditions, condition_lookup
from .models import Condition, FinalAnswerRecord, GenerationJob, GenerationRecord, ManifestEntry
from .reflection import build_stage_input, build_stage_instructions, stage_specs_for
from .utils import iter_json_files, read_json, sha256_text, write_json

LOGGER = logging.getLogger(__name__)


def build_cache_key(dataset: str, prompt_id: str, condition_id: str, generator_model: str, condition_text_hash: str) -> str:
    return sha256_text("::".join([dataset, prompt_id, condition_id, generator_model, condition_text_hash]))


def build_stage_cache_key(job_cache_key: str, stage_name: str, stage_input_text: str) -> str:
    return sha256_text("::".join([job_cache_key, stage_name, sha256_text(stage_input_text)]))


def trace_path_for_key(cache_key: str) -> Path:
    return config.TRACE_DIR / f"{cache_key}.json"


def final_answer_path_for_key(cache_key: str) -> Path:
    return config.FINAL_ANSWERS_DIR / f"{cache_key}.json"


def load_generation_records() -> list[GenerationRecord]:
    records: list[GenerationRecord] = []
    for path in iter_json_files(config.TRACE_DIR):
        payload = read_json(path)
        records.append(GenerationRecord.model_validate(payload))
    return records


def load_final_answer_records() -> list[FinalAnswerRecord]:
    records: list[FinalAnswerRecord] = []
    for path in iter_json_files(config.FINAL_ANSWERS_DIR):
        payload = read_json(path)
        records.append(FinalAnswerRecord.model_validate(payload))
    return records


def build_generation_jobs(
    manifest_entries: Iterable[ManifestEntry],
    generator_model: str,
    conditions: list[Condition] | None = None,
) -> list[GenerationJob]:
    selected_conditions = conditions or build_conditions()
    jobs: list[GenerationJob] = []
    for entry in manifest_entries:
        for condition in selected_conditions:
            condition_text_hash = sha256_text(
                "::".join(
                    [
                        condition.base_system_prompt,
                        condition.persona_prompt,
                        condition.constraint_prompt,
                        condition.reflection_mode,
                    ]
                )
            )
            jobs.append(
                GenerationJob(
                    prompt_id=entry.prompt_id,
                    prompt_text=entry.prompt_text,
                    reference_output=entry.reference_output,
                    generator_model=generator_model,
                    condition_id=condition.condition_id,
                    condition_text_hash=condition_text_hash,
                    cache_key=build_cache_key(
                        dataset="alpaca_eval",
                        prompt_id=entry.prompt_id,
                        condition_id=condition.condition_id,
                        generator_model=generator_model,
                        condition_text_hash=condition_text_hash,
                    ),
                )
            )
    return jobs


def missing_jobs(
    manifest_entries: Iterable[ManifestEntry],
    generator_model: str,
    conditions: list[Condition] | None = None,
) -> list[GenerationJob]:
    jobs = build_generation_jobs(manifest_entries, generator_model=generator_model, conditions=conditions)
    missing: list[GenerationJob] = []
    for job in jobs:
        final_path = final_answer_path_for_key(job.cache_key)
        if not final_path.exists():
            missing.append(job)
            continue
        record = FinalAnswerRecord.model_validate(read_json(final_path))
        if not record.final_response_text.strip():
            missing.append(job)
    return missing


def _serialize_usage(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {"raw": str(usage)}


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    pieces: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str):
                pieces.append(text_value)
    text = "\n".join(part for part in pieces if part).strip()
    if not text:
        raise ValueError("The OpenAI response did not contain any text output.")
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def _generate_response(
    client: OpenAI,
    model: str,
    instructions_text: str,
    input_text: str,
    max_output_tokens: int,
) -> Any:
    return client.responses.create(
        model=model,
        instructions=instructions_text,
        input=input_text,
        temperature=config.DEFAULT_TEMPERATURE,
        top_p=config.DEFAULT_TOP_P,
        max_output_tokens=max_output_tokens,
    )


def write_generation_index(records: list[GenerationRecord]) -> None:
    if not records:
        return
    df = pd.DataFrame([record.model_dump() for record in records]).sort_values(
        ["generator_model", "condition_id", "prompt_id", "stage_index"]
    )
    df.to_csv(config.ARTIFACTS_DIR / "alpaca_eval_stage_traces.csv", index=False)
    df.to_parquet(config.ARTIFACTS_DIR / "alpaca_eval_stage_traces.parquet", index=False)


def write_final_answer_index(records: list[FinalAnswerRecord]) -> None:
    if not records:
        return
    df = pd.DataFrame([record.model_dump() for record in records]).sort_values(
        ["generator_model", "condition_id", "prompt_id"]
    )
    df.to_csv(config.ARTIFACTS_DIR / "alpaca_eval_final_answers.csv", index=False)
    df.to_parquet(config.ARTIFACTS_DIR / "alpaca_eval_final_answers.parquet", index=False)


def run_alpaca_generation(
    generator_model: str,
    manifest_path: Path | None = None,
    limit_prompts: int | None = None,
    max_output_tokens: int = config.DEFAULT_MAX_OUTPUT_TOKENS,
    force: bool = False,
) -> list[FinalAnswerRecord]:
    config.ensure_directories()
    manifest_entries = read_manifest(manifest_path)
    if limit_prompts is not None:
        manifest_entries = manifest_entries[:limit_prompts]

    conditions = condition_lookup()
    jobs = build_generation_jobs(manifest_entries, generator_model=generator_model, conditions=list(conditions.values()))
    client = OpenAI()
    final_records: list[FinalAnswerRecord] = []

    for job in jobs:
        condition = conditions[job.condition_id]
        final_path = final_answer_path_for_key(job.cache_key)
        if final_path.exists() and not force:
            final_records.append(FinalAnswerRecord.model_validate(read_json(final_path)))
            continue

        previous_outputs: dict[str, str] = {}
        final_stage_record: GenerationRecord | None = None
        for stage in stage_specs_for(condition.reflection_mode):
            instructions_text = build_stage_instructions(condition, stage)
            input_text = build_stage_input(stage, job.prompt_text, previous_outputs)
            stage_cache_key = build_stage_cache_key(job.cache_key, stage.stage_name, input_text)
            trace_path = trace_path_for_key(stage_cache_key)
            if trace_path.exists() and not force:
                stage_record = GenerationRecord.model_validate(read_json(trace_path))
                if stage_record.status != "completed":
                    force_stage = True
                else:
                    previous_outputs[stage.stage_name] = stage_record.response_text
                    final_stage_record = stage_record
                    continue
            else:
                force_stage = True

            if force_stage:
                started = time.perf_counter()
                try:
                    response = _generate_response(
                        client=client,
                        model=job.generator_model,
                        instructions_text=instructions_text,
                        input_text=input_text,
                        max_output_tokens=max_output_tokens,
                    )
                    stage_record = GenerationRecord(
                        prompt_id=job.prompt_id,
                        condition_id=job.condition_id,
                        generator_model=job.generator_model,
                        stage_name=stage.stage_name,
                        stage_index=stage.stage_index,
                        input_text=input_text,
                        response_text=_extract_output_text(response),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        usage_json=_serialize_usage(response),
                        status="completed",
                        parent_stage=stage.parent_stage,
                        response_id=getattr(response, "id", None),
                        cache_key=stage_cache_key,
                    )
                except Exception as error:
                    LOGGER.exception(
                        "Generation failed for %s / %s / %s / %s",
                        job.generator_model,
                        job.condition_id,
                        job.prompt_id,
                        stage.stage_name,
                    )
                    stage_record = GenerationRecord(
                        prompt_id=job.prompt_id,
                        condition_id=job.condition_id,
                        generator_model=job.generator_model,
                        stage_name=stage.stage_name,
                        stage_index=stage.stage_index,
                        input_text=input_text,
                        response_text="",
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        usage_json={},
                        status="error",
                        parent_stage=stage.parent_stage,
                        response_id=None,
                        error_message=str(error),
                        cache_key=stage_cache_key,
                    )
                    write_json(trace_path, stage_record.model_dump())
                    final_stage_record = None
                    break

                write_json(trace_path, stage_record.model_dump())
                previous_outputs[stage.stage_name] = stage_record.response_text
                final_stage_record = stage_record

        if final_stage_record is None or final_stage_record.status != "completed":
            continue

        final_record = FinalAnswerRecord(
            prompt_id=job.prompt_id,
            condition_id=job.condition_id,
            generator_model=job.generator_model,
            persona_id=condition.persona_id,
            constraint_id=condition.constraint_id,
            reflection_mode=condition.reflection_mode,
            prompt_text=job.prompt_text,
            reference_output=job.reference_output,
            final_response_text=final_stage_record.response_text,
            final_stage_name=final_stage_record.stage_name,
            final_stage_index=final_stage_record.stage_index,
            source_trace_cache_key=final_stage_record.cache_key,
            cache_key=job.cache_key,
        )
        write_json(final_path, final_record.model_dump())
        final_records.append(final_record)

    write_generation_index(load_generation_records())
    write_final_answer_index(load_final_answer_records())
    return final_records


def validate_generation_matrix(
    generator_model: str,
    manifest_path: Path | None = None,
    limit_prompts: int | None = None,
) -> list[str]:
    manifest_entries = read_manifest(manifest_path)
    if limit_prompts is not None:
        manifest_entries = manifest_entries[:limit_prompts]
    missing = missing_jobs(manifest_entries, generator_model=generator_model)
    return [f"{job.generator_model}:{job.condition_id}:{job.prompt_id}" for job in missing]
