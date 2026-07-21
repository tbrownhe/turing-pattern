from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
ENGINE_VERSION = "2.0.0"


@dataclass(frozen=True, slots=True)
class ControlValues:
    F1: float
    F2: float
    K1: float
    K2: float
    Du1: float
    Du2: float
    Dv1: float
    Dv2: float

    def __post_init__(self) -> None:
        ranges = {
            "F1": (self.F1, 0.0, 0.1),
            "F2": (self.F2, 0.0, 0.1),
            "K1": (self.K1, 0.0, 0.1),
            "K2": (self.K2, 0.0, 0.1),
            "Du1": (self.Du1, 0.0, 1.0),
            "Du2": (self.Du2, 0.0, 1.0),
            "Dv1": (self.Dv1, 0.0, 1.0),
            "Dv2": (self.Dv2, 0.0, 1.0),
        }
        for name, (value, minimum, maximum) in ranges.items():
            if not np.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(
                    f"{name} must be finite and between {minimum} and {maximum}"
                )

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> ControlValues:
        return cls(**{name: float(values[name]) for name in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, float]:
        return {
            "F1": self.F1,
            "F2": self.F2,
            "K1": self.K1,
            "K2": self.K2,
            "Du1": self.Du1,
            "Du2": self.Du2,
            "Dv1": self.Dv1,
            "Dv2": self.Dv2,
        }


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    width: int
    height: int
    controls: ControlValues
    seed: int = 0
    dtype: str = "float32"
    boundary: str = "periodic"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if not 0 <= self.seed <= 4_294_967_295:
            raise ValueError("seed must fit in an unsigned 32-bit integer")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        if self.boundary != "periodic":
            raise ValueError("only periodic boundaries are currently supported")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclass(slots=True)
class ParameterFields:
    feed: FloatArray
    kill: FloatArray
    diffusion_u: FloatArray
    diffusion_v: FloatArray

    @property
    def nbytes(self) -> int:
        return sum(
            field.nbytes
            for field in (self.feed, self.kill, self.diffusion_u, self.diffusion_v)
        )


@dataclass(slots=True)
class SimulationState:
    u: FloatArray
    v: FloatArray
    laplacian_u: FloatArray
    laplacian_v: FloatArray
    reaction: FloatArray
    iteration: int = 0

    @property
    def concentration_nbytes(self) -> int:
        return self.u.nbytes + self.v.nbytes

    @property
    def working_nbytes(self) -> int:
        return sum(
            field.nbytes
            for field in (
                self.u,
                self.v,
                self.laplacian_u,
                self.laplacian_v,
                self.reaction,
            )
        )
