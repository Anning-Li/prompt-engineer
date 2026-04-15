from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ManifestEntry(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = "alpaca_eval"
    prompt_id: str
    source_index: int
    prompt_text: str
    reference_output: str


class Condition(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    condition_id: str
    persona_id: Literal["A", "B", "C"]
    constraint_id: Literal["C1", "C2", "C3"]
    reflection_mode: Literal["R0", "R1", "R2", "R3"]
    display_row: str
    display_column: str
    display_order: int
    base_system_prompt: str
    persona_prompt: str
    constraint_prompt: str


class GenerationJob(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = "alpaca_eval"
    prompt_id: str
    prompt_text: str
    reference_output: str
    generator_model: str
    condition_id: str
    condition_text_hash: str
    cache_key: str


class GenerationRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = "alpaca_eval"
    prompt_id: str
    condition_id: str
    generator_model: str
    stage_name: str
    stage_index: int
    input_text: str
    response_text: str
    latency_ms: int
    usage_json: dict[str, Any]
    status: Literal["completed", "error"]
    parent_stage: str | None = None
    response_id: str | None = None
    error_message: str | None = None
    cache_key: str
    created_at: str = Field(default_factory=utc_now_iso)


class FinalAnswerRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = "alpaca_eval"
    prompt_id: str
    condition_id: str
    generator_model: str
    persona_id: str
    constraint_id: str
    reflection_mode: str
    prompt_text: str
    reference_output: str
    final_response_text: str
    final_stage_name: str
    final_stage_index: int
    source_trace_cache_key: str
    cache_key: str
    created_at: str = Field(default_factory=utc_now_iso)


class CorrectnessJudgmentRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = "alpaca_eval"
    prompt_id: str
    condition_id: str
    generator_model: str
    persona_id: str
    constraint_id: str
    reflection_mode: str
    score: int = Field(ge=1, le=5)
    reason: str
    judge_model: str
    cache_key: str
    created_at: str = Field(default_factory=utc_now_iso)


class AggregateMetricRecord(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    dataset: str = "alpaca_eval"
    generator_model: str
    persona_id: str
    constraint_id: str
    reflection_mode: str
    condition_id: str
    mean_correctness_score: float
    n_prompts: int
    run_id: str
