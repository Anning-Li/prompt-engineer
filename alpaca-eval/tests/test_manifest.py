from pathlib import Path

from steering_eval import config
from steering_eval.alpaca_dataset import build_manifest


def test_manifest_is_stable(monkeypatch, tmp_path: Path) -> None:
    fake_examples = [
        {
            "source_index": index,
            "prompt_text": f"prompt {index}",
            "reference_output": f"reference {index}",
        }
        for index in range(200)
    ]
    manifest_path = tmp_path / "manifest.json"

    monkeypatch.setattr("steering_eval.alpaca_dataset.load_source_examples", lambda: fake_examples)

    first = build_manifest(sample_size=50, seed=config.DEFAULT_SEED, output_path=manifest_path, force=True)
    second = build_manifest(sample_size=50, seed=config.DEFAULT_SEED, output_path=manifest_path, force=False)

    assert len(first) == 50
    assert [entry.prompt_id for entry in first] == [entry.prompt_id for entry in second]
    assert [entry.source_index for entry in first] == sorted(entry.source_index for entry in first)
