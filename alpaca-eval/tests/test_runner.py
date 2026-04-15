from pathlib import Path
from types import SimpleNamespace

from steering_eval import config
from steering_eval.conditions import build_conditions
from steering_eval.generation import (
    build_generation_jobs,
    build_stage_cache_key,
    final_answer_path_for_key,
    run_alpaca_generation,
    trace_path_for_key,
)
from steering_eval.models import FinalAnswerRecord, GenerationRecord, ManifestEntry
from steering_eval.reflection import build_stage_input, stage_specs_for
from steering_eval.utils import write_json


def _fake_manifest(size: int = 50) -> list[ManifestEntry]:
    return [
        ManifestEntry(
            prompt_id=f"alpaca_{index:04d}",
            source_index=index,
            prompt_text=f"prompt {index}",
            reference_output=f"reference {index}",
        )
        for index in range(size)
    ]


def test_run_schedule_has_full_cartesian_count() -> None:
    jobs = build_generation_jobs(_fake_manifest(), generator_model="m1", conditions=build_conditions())
    assert len(jobs) == 50 * 36


class _CountingResponses:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(str(kwargs["input"]))
        stage_number = len(self.calls)
        return SimpleNamespace(
            id=f"resp_{stage_number}",
            output_text=f"generated stage {stage_number}",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 10}),
        )


class _CountingOpenAI:
    def __init__(self, responses: _CountingResponses) -> None:
        self.responses = responses


def test_stage_cache_resume_skips_completed_stages(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(config, "MANIFESTS_DIR", tmp_path / "data" / "manifests")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts" / "alpaca_eval")
    monkeypatch.setattr(config, "TRACE_DIR", tmp_path / "artifacts" / "alpaca_eval" / "stage_traces")
    monkeypatch.setattr(config, "FINAL_ANSWERS_DIR", tmp_path / "artifacts" / "alpaca_eval" / "final_answers")
    monkeypatch.setattr(config, "SCORES_DIR", tmp_path / "artifacts" / "alpaca_eval" / "scores")
    monkeypatch.setattr(config, "JUDGMENTS_DIR", tmp_path / "artifacts" / "alpaca_eval" / "judgments")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "artifacts" / "alpaca_eval" / "reports")
    monkeypatch.setattr(config, "PLOTS_DIR", tmp_path / "artifacts" / "alpaca_eval" / "reports" / "plots")
    config.ensure_directories()

    condition = next(condition for condition in build_conditions() if condition.condition_id == "a_c1_r2")
    monkeypatch.setattr("steering_eval.generation.condition_lookup", lambda: {condition.condition_id: condition})

    manifest = _fake_manifest(1)
    manifest_path = tmp_path / "data" / "manifests" / "manifest.json"
    write_json(manifest_path, [entry.model_dump() for entry in manifest])

    job = build_generation_jobs(manifest, generator_model="m1", conditions=[condition])[0]
    draft_stage = stage_specs_for("R2")[0]
    draft_input = build_stage_input(draft_stage, job.prompt_text, {})
    draft_cache_key = build_stage_cache_key(job.cache_key, draft_stage.stage_name, draft_input)
    write_json(
        trace_path_for_key(draft_cache_key),
        GenerationRecord(
            prompt_id=job.prompt_id,
            condition_id=job.condition_id,
            generator_model=job.generator_model,
            stage_name="draft",
            stage_index=0,
            input_text=draft_input,
            response_text="cached draft",
            latency_ms=3,
            usage_json={"total_tokens": 6},
            status="completed",
            cache_key=draft_cache_key,
        ).model_dump(),
    )

    responses = _CountingResponses()
    monkeypatch.setattr("steering_eval.generation.OpenAI", lambda: _CountingOpenAI(responses))

    records = run_alpaca_generation(
        generator_model="m1",
        manifest_path=manifest_path,
        max_output_tokens=32,
        force=False,
    )

    assert len(responses.calls) == 2
    assert len(records) == 1
    final_path = final_answer_path_for_key(job.cache_key)
    assert final_path.exists()
    final_record = FinalAnswerRecord.model_validate_json(final_path.read_text())
    assert final_record.final_stage_name == "revise_final"
