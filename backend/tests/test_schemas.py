import json

import pytest
from pydantic import ValidationError

from app.api.schemas import ControlsMessage, StartMessage, parse_client_message

VALID_CONTROLS = {
    "F1": 0.04,
    "F2": 0.08,
    "K1": 0.056,
    "K2": 0.074,
    "Du1": 0.7,
    "Du2": 0.7,
    "Dv1": 0.25,
    "Dv2": 0.25,
}


def parse(payload):
    return parse_client_message(json.dumps(payload))


def test_start_message_is_strict_and_versioned():
    message = parse(
        {
            "type": "start",
            "protocol_version": 1,
            "controls": VALID_CONTROLS,
            "seed": 42,
        }
    )

    assert isinstance(message, StartMessage)
    assert message.seed == 42


def test_control_message_round_trips_displayed_values():
    message = parse({"type": "controls", "controls": VALID_CONTROLS})

    assert isinstance(message, ControlsMessage)
    assert message.controls.F1 == 0.04


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "start", "controls": VALID_CONTROLS, "shape": [50_000, 50_000]},
        {"type": "start", "controls": {**VALID_CONTROLS, "F1": -0.01}},
        {"type": "start", "controls": {**VALID_CONTROLS, "Dv2": 2.0}},
        {"type": "start", "controls": {**VALID_CONTROLS, "F1": float("nan")}},
        {"type": "start", "controls": {**VALID_CONTROLS, "K1": float("inf")}},
        {"type": "start", "controls": {**VALID_CONTROLS, "F1": "0.04"}},
        {"type": "unknown", "controls": VALID_CONTROLS},
        {"type": "controls", "controls": {"F1": 0.04}},
    ],
)
def test_invalid_or_resource_controlling_messages_are_rejected(payload):
    with pytest.raises(ValidationError):
        parse(payload)
