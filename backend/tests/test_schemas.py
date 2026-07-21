import json

import pytest
from pydantic import ValidationError

from app.api.schemas import (
    ControlsMessage,
    RenderPlanRequest,
    StartMessage,
    StepMessage,
    TimeStudyRequest,
    parse_client_message,
)

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


def test_control_message_round_trips_displayed_values_and_revision():
    message = parse({"type": "controls", "controls": VALID_CONTROLS, "revision": 17})

    assert isinstance(message, ControlsMessage)
    assert message.controls.F1 == 0.04
    assert message.revision == 17

    with pytest.raises(ValidationError):
        parse({"type": "controls", "controls": VALID_CONTROLS, "revision": 0})


def test_step_message_has_no_client_controlled_work_size():
    message = parse({"type": "step"})

    assert isinstance(message, StepMessage)

    with pytest.raises(ValidationError):
        parse({"type": "step", "iterations": 1_000_000})


def test_render_plans_and_time_studies_are_strict_and_bounded():
    plan = RenderPlanRequest.model_validate(
        {
            "controls": VALID_CONTROLS,
            "seed": 42,
            "width": 6.0,
            "height": 6.0,
            "unit": "in",
            "quality": "studio",
            "feature_scale": 1.0,
            "development_steps": 5000,
            "framing": "crop",
        }
    )
    study = TimeStudyRequest.model_validate(
        {"controls": VALID_CONTROLS, "seed": 42, "checkpoints": [100, 500]}
    )

    assert plan.development_steps == 5000
    assert study.checkpoints == [100, 500]

    with pytest.raises(ValidationError):
        TimeStudyRequest.model_validate(
            {"controls": VALID_CONTROLS, "checkpoints": [500, 100]}
        )
    with pytest.raises(ValidationError):
        RenderPlanRequest.model_validate(
            {
                **plan.model_dump(),
                "development_steps": 1_000_000,
            }
        )


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
