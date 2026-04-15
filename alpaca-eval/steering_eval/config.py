from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
MANIFESTS_DIR = DATA_DIR / "manifests"
ARTIFACTS_DIR = ROOT_DIR / "artifacts" / "alpaca_eval"
TRACE_DIR = ARTIFACTS_DIR / "stage_traces"
FINAL_ANSWERS_DIR = ARTIFACTS_DIR / "final_answers"
SCORES_DIR = ARTIFACTS_DIR / "scores"
JUDGMENTS_DIR = ARTIFACTS_DIR / "judgments"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

DEFAULT_SEED = 20260414
DEFAULT_SAMPLE_SIZE = 50
DEFAULT_MAX_OUTPUT_TOKENS = int(os.getenv("STEERING_EVAL_MAX_OUTPUT_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("STEERING_EVAL_TEMPERATURE", "0"))
DEFAULT_TOP_P = float(os.getenv("STEERING_EVAL_TOP_P", "1"))

DEFAULT_CORRECTNESS_JUDGE_MODEL = os.getenv("STEERING_EVAL_CORRECTNESS_JUDGE_MODEL", "gpt-4.1")
ALPACA_EVAL_SOURCE_URL = os.getenv(
    "STEERING_EVAL_ALPACA_SOURCE_URL",
    "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/main/alpaca_eval.json",
)


def manifest_path(sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = DEFAULT_SEED) -> Path:
    return MANIFESTS_DIR / f"alpaca_eval_manifest_seed{seed}_n{sample_size}.json"


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        MANIFESTS_DIR,
        ARTIFACTS_DIR,
        TRACE_DIR,
        FINAL_ANSWERS_DIR,
        SCORES_DIR,
        JUDGMENTS_DIR,
        REPORTS_DIR,
        PLOTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
