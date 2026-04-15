from steering_eval.conditions import build_conditions
from steering_eval.reflection import build_stage_instructions, build_stage_input, stage_specs_for


def test_reflection_orchestration_has_expected_stage_counts() -> None:
    assert [stage.stage_name for stage in stage_specs_for("R0")] == ["final"]
    assert [stage.stage_name for stage in stage_specs_for("R1")] == ["draft", "checklist_final"]
    assert [stage.stage_name for stage in stage_specs_for("R2")] == ["draft", "critique", "revise_final"]
    assert [stage.stage_name for stage in stage_specs_for("R3")] == [
        "draft",
        "critique",
        "revise_final",
        "calibrate_final",
    ]


def test_prompt_assembly_uses_expected_blocks() -> None:
    condition = build_conditions()[0]
    draft_stage = stage_specs_for("R2")[0]
    critique_stage = stage_specs_for("R2")[1]
    revise_stage = stage_specs_for("R2")[2]

    instructions = build_stage_instructions(condition, draft_stage)
    assert "You are a helpful assistant." in instructions
    assert "Persona: Default assistant." in instructions
    assert "Constraint priority: No additional constraints" in instructions
    assert "Dataset 1-shot" not in instructions

    draft_input = build_stage_input(draft_stage, "Solve 2+2.", {})
    assert "Dataset Prompt:\nSolve 2+2." in draft_input
    assert "Previous Output" not in draft_input

    critique_input = build_stage_input(critique_stage, "Solve 2+2.", {"draft": "It is 4."})
    assert "Previous Output:\nIt is 4." in critique_input

    revise_input = build_stage_input(
        revise_stage,
        "Solve 2+2.",
        {"draft": "It is 4.", "critique": "- Good but too wordy."},
    )
    assert "Previous Output:\nIt is 4." in revise_input
    assert "Critique:\n- Good but too wordy." in revise_input
