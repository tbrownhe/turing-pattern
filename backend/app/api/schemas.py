from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

FeedRate = Annotated[float, Field(ge=0.0, le=0.1, allow_inf_nan=False, strict=True)]
KillRate = Annotated[float, Field(ge=0.0, le=0.1, allow_inf_nan=False, strict=True)]
DiffusionRate = Annotated[
    float, Field(ge=0.0, le=1.0, allow_inf_nan=False, strict=True)
]
Seed = Annotated[int, Field(ge=0, le=4_294_967_295, strict=True)]
DevelopmentStep = Annotated[int, Field(ge=100, le=20_000, strict=True)]
ControlRevision = Annotated[int, Field(ge=1, le=2_147_483_647, strict=True)]
PhysicalDimension = Annotated[
    float, Field(gt=0.0, le=100.0, allow_inf_nan=False, strict=True)
]
FeatureScale = Annotated[float, Field(ge=0.5, le=2.0, allow_inf_nan=False, strict=True)]


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
    revision: ControlRevision | None = None


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


class TimeStudyRequest(StrictModel):
    controls: Controls
    seed: Seed = 0
    checkpoints: list[DevelopmentStep] = Field(min_length=2, max_length=6)

    @field_validator("checkpoints")
    @classmethod
    def checkpoints_are_unique_and_increasing(cls, values: list[int]) -> list[int]:
        if values != sorted(set(values)):
            raise ValueError("checkpoints must be unique and strictly increasing")
        return values


class RenderPlanRequest(StrictModel):
    controls: Controls
    seed: Seed = 0
    width: PhysicalDimension
    height: PhysicalDimension
    unit: Literal["in", "cm"]
    quality: Literal["draft", "studio", "fine"]
    feature_scale: FeatureScale
    development_steps: DevelopmentStep
    framing: Literal["crop", "fit", "extend"] = "crop"

    @field_validator("feature_scale")
    @classmethod
    def feature_scale_is_supported(cls, value: float) -> float:
        if value not in {0.5, 1.0, 2.0}:
            raise ValueError("feature_scale must be 0.5, 1.0, or 2.0")
        return value


def parse_client_message(raw: str) -> ClientMessage:
    return client_message_adapter.validate_json(raw)
