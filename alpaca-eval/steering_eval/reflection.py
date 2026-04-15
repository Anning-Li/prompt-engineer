from __future__ import annotations

from dataclasses import dataclass

from .models import Condition


@dataclass(frozen=True)
class StageSpec:
    stage_name: str
    stage_index: int
    parent_stage: str | None
    role_prompt: str


def stage_specs_for(reflection_mode: str) -> list[StageSpec]:
    draft_stage = StageSpec(
        stage_name="draft",
        stage_index=0,
        parent_stage=None,
        role_prompt=(
            "Current task role: Draft writer.\n"
            "Produce an initial answer to the user request. Do not include commentary about drafting."
        ),
    )
    critique_stage = StageSpec(
        stage_name="critique",
        stage_index=1,
        parent_stage="draft",
        role_prompt=(
            "Current task role: Critic.\n"
            "Critique the draft for the following:\n"
            "- factual weakness or unsupported claims\n"
            "- instruction-following or formatting failures\n"
            "- unnecessary refusal\n"
            "- insufficient refusal when caution is required\n"
            "- excessive hedging\n"
            "- excessive verbosity\n"
            "- style inconsistency with the assigned persona\n"
            "- mismatch with the assigned constraint strength\n"
            "Return concise bullet points only. Be specific and actionable."
        ),
    )
    if reflection_mode == "R0":
        return [
            StageSpec(
                stage_name="final",
                stage_index=0,
                parent_stage=None,
                role_prompt=(
                    "Current task role: Final responder.\n"
                    "Produce the best possible final answer to the user request in one pass."
                ),
            )
        ]
    if reflection_mode == "R1":
        return [
            draft_stage,
            StageSpec(
                stage_name="checklist_final",
                stage_index=1,
                parent_stage="draft",
                role_prompt=(
                    "Current task role: Checklist reviewer and final responder.\n"
                    "Evaluate the draft internally using this checklist:\n"
                    "- Is the answer responsive to the user's request?\n"
                    "- Is it factually supported or appropriately uncertain?\n"
                    "- Does it follow the requested format?\n"
                    "- Is it unnecessarily verbose?\n"
                    "- Is it unnecessarily hedged?\n"
                    "- If it refuses or withholds, is that actually necessary?\n"
                    "Review your response against these three principles: "
                    "(1) Is it genuinely helpful? (2) Is it honest and accurate? "
                    "(3) Does it avoid unnecessary harm?\n"
                    "Then produce the improved final answer. Output the final answer only."
                ),
            ),
        ]
    if reflection_mode == "R2":
        return [
            draft_stage,
            critique_stage,
            StageSpec(
                stage_name="revise_final",
                stage_index=2,
                parent_stage="critique",
                role_prompt=(
                    "Current task role: Final reviser.\n"
                    "Use the critique to produce the best final answer. Resolve identified issues if possible. "
                    "Output the final answer only."
                ),
            ),
        ]
    if reflection_mode == "R3":
        return [
            draft_stage,
            critique_stage,
            StageSpec(
                stage_name="revise_final",
                stage_index=2,
                parent_stage="critique",
                role_prompt=(
                    "Current task role: Final reviser.\n"
                    "Use the critique to produce the best final answer. Resolve identified issues if possible. "
                    "Output the final answer only."
                ),
            ),
            StageSpec(
                stage_name="calibrate_final",
                stage_index=3,
                parent_stage="revise_final",
                role_prompt=(
                    "Current task role: Final calibrator.\n"
                    "Check the current answer for the following:\n"
                    "- If it makes strong factual claims, ensure confidence matches evidence.\n"
                    "- If it refuses or withholds, verify that refusal is actually necessary.\n"
                    "- If it is safe but unhelpful, make it more helpful while preserving the constraints.\n"
                    "- If it is overly verbose, shorten it.\n"
                    "- If it is overly hedged, make it more direct without overstating certainty.\n"
                    "- Internally rate confidence from 1-5. If confidence is below 3, explicitly state what is "
                    "uncertain or whether you should abstain.\n"
                    "If the current answer is already good, keep changes minimal. Output the final answer only."
                ),
            ),
        ]
    raise ValueError(f"Unsupported reflection mode: {reflection_mode}")


def build_stage_instructions(condition: Condition, stage: StageSpec) -> str:
    return "\n\n".join(
        [
            condition.base_system_prompt,
            condition.persona_prompt,
            condition.constraint_prompt,
            stage.role_prompt,
        ]
    )


def build_stage_input(stage: StageSpec, dataset_prompt: str, previous_outputs: dict[str, str]) -> str:
    sections = [f"Dataset Prompt:\n{dataset_prompt}"]
    if stage.stage_name == "checklist_final":
        sections.append(f"Previous Output:\n{previous_outputs['draft']}")
    elif stage.stage_name == "critique":
        sections.append(f"Previous Output:\n{previous_outputs['draft']}")
    elif stage.stage_name == "revise_final":
        sections.append(f"Previous Output:\n{previous_outputs['draft']}")
        sections.append(f"Critique:\n{previous_outputs['critique']}")
    elif stage.stage_name == "calibrate_final":
        sections.append(f"Current Answer:\n{previous_outputs['revise_final']}")
        sections.append(f"Critique:\n{previous_outputs['critique']}")
    return "\n\n".join(sections)
