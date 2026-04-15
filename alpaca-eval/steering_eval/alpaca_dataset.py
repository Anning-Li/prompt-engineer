from __future__ import annotations

import random
from pathlib import Path

from datasets import load_dataset

from . import config
from .models import ManifestEntry
from .utils import read_json, write_json


def load_source_examples() -> list[dict[str, str]]:
    try:
        dataset = load_dataset("json", data_files=config.ALPACA_EVAL_SOURCE_URL, split="train")
    except Exception as first_error:
        try:
            dataset = load_dataset("tatsu-lab/alpaca_eval", split="eval")
        except Exception as second_error:
            raise RuntimeError(
                "Unable to load the AlpacaEval source dataset. "
                "Install dependencies and ensure network access is available for the first manifest build."
            ) from second_error if second_error else first_error

    examples: list[dict[str, str]] = []
    for index, row in enumerate(dataset):
        prompt_text = row.get("instruction") or ""
        reference_output = row.get("output") or ""
        if not prompt_text or not reference_output:
            continue
        examples.append(
            {
                "source_index": index,
                "prompt_text": prompt_text,
                "reference_output": reference_output,
            }
        )
    return examples


def read_manifest(path: Path | None = None) -> list[ManifestEntry]:
    manifest_file = path or config.manifest_path()
    payload = read_json(manifest_file)
    return [ManifestEntry.model_validate(item) for item in payload]


def build_manifest(
    sample_size: int = config.DEFAULT_SAMPLE_SIZE,
    seed: int = config.DEFAULT_SEED,
    output_path: Path | None = None,
    force: bool = False,
) -> list[ManifestEntry]:
    config.ensure_directories()
    manifest_file = output_path or config.manifest_path(sample_size=sample_size, seed=seed)
    if manifest_file.exists() and not force:
        return read_manifest(manifest_file)

    source_examples = load_source_examples()
    if len(source_examples) < sample_size:
        raise ValueError(
            f"Requested {sample_size} AlpacaEval prompts, but only {len(source_examples)} are available."
        )

    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(source_examples)), sample_size))
    manifest = [
        ManifestEntry(
            prompt_id=f"alpaca_{source_examples[source_index]['source_index']:04d}",
            source_index=source_examples[source_index]["source_index"],
            prompt_text=source_examples[source_index]["prompt_text"],
            reference_output=source_examples[source_index]["reference_output"],
        )
        for source_index in selected_indices
    ]
    write_json(manifest_file, [entry.model_dump() for entry in manifest])
    return manifest
