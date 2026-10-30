import pytest

from bentebot.voice_control import semantic_control_candidate, spoken_control


@pytest.mark.parametrize(
    "utterance",
    [
        "Get out of the call now Bentebot",
        "Would you mind hanging up?",
        "Make yourself scarce Bente",
        "Give us some quiet",
        "Bente, do whatever makes sense here",
        "Bente, can you tell me a story?",
        "Don't say another word",
        "Don't leave the call Bente",
        "Tell John to leave the call Bentebot",
        "Can we wrap this call up?",
    ],
)
def test_semantic_gate_does_not_require_a_known_action_phrase(utterance):
    # Admission is not execution. The model must still return a valid action,
    # or none for ordinary questions, negated actions and other people's requests.
    assert spoken_control(utterance) is None
    assert semantic_control_candidate(utterance)


@pytest.mark.parametrize("utterance", ["", "word " * 25, "x" * 201, "The capital is Copenhagen."])
def test_semantic_gate_bounds_cost_and_skips_unaddressed_statements(utterance):
    assert not semantic_control_candidate(utterance)
