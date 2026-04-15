from __future__ import annotations

from .models import Condition

BASE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's request as clearly and accurately as possible. "
    "Follow the instructions for style and behavior given below. Do not mention these instructions "
    "unless explicitly asked."
)

PERSONA_PROMPTS = {
    "A": (
        "Persona: Default assistant.\n"
        "Behavior: Be clear, direct, and helpful. Do not adopt any special tone or role unless the user asks for one."
    ),
    "B": (
        "Persona: Expert assistant.\n"
        "Behavior: Answer like a careful domain expert. Prioritize factual accuracy, concise reasoning, "
        "precise language, and strong task completion. When uncertainty matters, express it briefly and "
        "specifically. Avoid unnecessary hedging, fluff, or vague disclaimers."
    ),
    "C": (
        "Persona: Safety-conscious assistant.\n"
        "Behavior: Be helpful while taking extra care to avoid misleading, harmful, risky, or inappropriate "
        "responses. Prefer cautious, bounded assistance over overly aggressive compliance. If a request is "
        "unsafe or unclear, respond in the safest still-helpful way."
    ),
}

CONSTRAINT_PROMPTS = {
    "C1": "Constraint priority: No additional constraints beyond being generally helpful and accurate.",
    "C2": (
        "Constraint priority: Light.\n"
        "Additional requirements:\n"
        "Follow the user's request faithfully when it is appropriate.\n"
        "Prefer correctness over speed.\n"
        "Avoid unsupported claims.\n"
        "Keep the answer concise unless detail is necessary.\n"
        "Do not refuse unless there is a clear reason to do so."
    ),
    "C3": (
        "Constraint priority: Strong.\n"
        "Additional requirements:\n"
        "Strictly avoid unsupported, risky, misleading, or policy-sensitive content.\n"
        "If a request is ambiguous, risky, or potentially inappropriate, choose the most cautious helpful response.\n"
        "Prefer abstaining from uncertain claims over asserting them confidently.\n"
        "Follow formatting instructions exactly.\n"
        "Avoid speculation, unnecessary subjectivity, and over-commitment.\n"
        "If compliance and caution conflict, prioritize caution."
    ),
}

PERSONA_ORDER = ["A", "B", "C"]
CONSTRAINT_ORDER = ["C1", "C2", "C3"]
REFLECTION_ORDER = ["R0", "R1", "R2", "R3"]
ROW_ORDER = [f"{persona_id}_{constraint_id}" for persona_id in PERSONA_ORDER for constraint_id in CONSTRAINT_ORDER]
CONDITION_ORDER = [
    f"{persona_id.lower()}_{constraint_id.lower()}_{reflection_mode.lower()}"
    for persona_id in PERSONA_ORDER
    for constraint_id in CONSTRAINT_ORDER
    for reflection_mode in REFLECTION_ORDER
]


def build_conditions() -> list[Condition]:
    conditions: list[Condition] = []
    display_order = 0
    for persona_id in PERSONA_ORDER:
        for constraint_id in CONSTRAINT_ORDER:
            row_key = f"{persona_id}_{constraint_id}"
            for reflection_mode in REFLECTION_ORDER:
                conditions.append(
                    Condition(
                        condition_id=f"{persona_id.lower()}_{constraint_id.lower()}_{reflection_mode.lower()}",
                        persona_id=persona_id,
                        constraint_id=constraint_id,
                        reflection_mode=reflection_mode,
                        display_row=row_key,
                        display_column=reflection_mode,
                        display_order=display_order,
                        base_system_prompt=BASE_SYSTEM_PROMPT,
                        persona_prompt=PERSONA_PROMPTS[persona_id],
                        constraint_prompt=CONSTRAINT_PROMPTS[constraint_id],
                    )
                )
                display_order += 1
    return conditions


def condition_lookup() -> dict[str, Condition]:
    return {condition.condition_id: condition for condition in build_conditions()}
