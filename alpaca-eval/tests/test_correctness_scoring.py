from steering_eval.alpaca_scoring import parse_correctness_judgment_response


def test_parse_correctness_judgment_response_accepts_wrapped_json() -> None:
    payload = parse_correctness_judgment_response(
        "```json\n{\"score\": 4, \"reason\": \"Mostly correct with a minor omission.\"}\n```"
    )
    assert payload["score"] == 4
    assert payload["reason"] == "Mostly correct with a minor omission."
