from steering_eval.conditions import CONDITION_ORDER, build_conditions


def test_conditions_are_stable() -> None:
    conditions = build_conditions()
    assert len(conditions) == 36
    assert [condition.condition_id for condition in conditions] == CONDITION_ORDER
    assert [condition.display_order for condition in conditions] == list(range(36))
    assert conditions[0].condition_id == "a_c1_r0"
    assert conditions[0].display_row == "A_C1"
    assert conditions[0].display_column == "R0"
    assert conditions[-1].condition_id == "c_c3_r3"
    assert conditions[-1].display_row == "C_C3"
    assert conditions[-1].display_column == "R3"
