from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from steering_eval import config
from steering_eval.alpaca_scoring import score_alpaca
from steering_eval.generation import run_alpaca_generation
from steering_eval.models import ManifestEntry
from steering_eval.reporting import build_pivot_table, report_alpaca
from steering_eval.utils import write_json


class _FakeGenerationResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs: object) -> object:
        self.calls += 1
        return SimpleNamespace(
            id=f"resp_gen_{self.calls}",
            output_text=f"Synthetic answer {self.calls} for {kwargs['input']}",
            usage=SimpleNamespace(model_dump=lambda: {"total_tokens": 12}),
        )


class _FakeGenerationOpenAI:
    def __init__(self) -> None:
        self.responses = _FakeGenerationResponses()


def test_smoke_pipeline(monkeypatch, tmp_path: Path) -> None:
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

    manifest = [
        ManifestEntry(
            prompt_id=f"alpaca_{index:04d}",
            source_index=index,
            prompt_text=f"prompt {index}",
            reference_output=f"reference {index}",
        )
        for index in range(3)
    ]
    manifest_path = tmp_path / "data" / "manifests" / "manifest.json"
    write_json(manifest_path, [entry.model_dump() for entry in manifest])

    monkeypatch.setattr("steering_eval.generation.OpenAI", _FakeGenerationOpenAI)
    final_answers = run_alpaca_generation(
        generator_model="gpt-test",
        manifest_path=manifest_path,
        max_output_tokens=32,
        force=True,
    )
    assert len(final_answers) == 3 * 36

    monkeypatch.setattr(
        "steering_eval.alpaca_scoring._judge_response",
        lambda client, judge_model, input_text: SimpleNamespace(
            output_text='{"score": 4, "reason": "Mostly correct."}'
        ),
    )
    monkeypatch.setattr("steering_eval.alpaca_scoring.OpenAI", lambda: object())
    score_df = score_alpaca(
        generator_model="gpt-test",
        manifest_path=manifest_path,
        judge_model="judge-test",
        run_id="smoke",
        force=True,
    )
    assert len(score_df) == 36
    pivot_df = build_pivot_table(score_df)
    assert pivot_df.shape == (9, 6)

    outputs = report_alpaca(run_id="smoke")
    assert outputs["report"].exists()
    assert outputs["heatmap"].exists()
    reported_pivot = pd.read_csv(outputs["pivot_csv"])
    assert reported_pivot.shape == (9, 6)
