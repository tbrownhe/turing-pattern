from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

FeedRate = Annotated[float, Field(ge=0.0, le=0.1, allow_inf_nan=False, strict=True)]
KillRate = Annotated[float, Field(ge=0.0, le=0.1, allow_inf_nan=False, strict=True)]
DiffusionRate = Annotated[
    float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
]
Seed = Annotated[int, Field(ge=0, le=4_294_967_295, strict=True)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Controls(StrictModel):
    F1: FeedRate
    F2: FeedRate
    K1: KillRate
    K2: KillRate
    Du1: DiffusionRate
    Du2: DiffusionRate
    Dv1: DiffusionRate
    Dv2: DiffusionRate


class StartMessage(StrictModel):
    type: Literal["start"]
    protocol_version: Literal[1] = 1
    controls: Controls
    seed: Seed = 0


class ControlsMessage(StrictModel):
    type: Literal["controls"]
    controls: Controls


class ResetMessage(StrictModel):
    type: Literal["reset"]
    seed: Seed


class PerturbMessage(StrictModel):
    type: Literal["perturb"]
    noise: Annotated[float, Field(gt=0.0, le=1.0, allow_inf_nan=False, strict=True)] = (
        0.25
    )


class PauseMessage(StrictModel):
    type: Literal["pause"]


class ResumeMessage(StrictModel):
    type: Literal["resume"]


class StepMessage(StrictModel):
    type: Literal["step"]


ClientMessage = Annotated[
    StartMessage
    | ControlsMessage
    | ResetMessage
    | PerturbMessage
    | PauseMessage
    | ResumeMessage
    | StepMessage,
    Field(discriminator="type"),
]
client_message_adapter: TypeAdapter[ClientMessage] = TypeAdapter(ClientMessage)


class RenderRequest(StrictModel):
    controls: Controls
    seed: Seed = 0


def parse_client_message(raw: str) -> ClientMessage:
    return client_message_adapter.validate_json(raw)
