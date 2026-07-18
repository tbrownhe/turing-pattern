from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
from scipy.ndimage import convolve

from app.core.models import (
    ControlValues,
    FloatArray,
    ParameterFields,
    SimulationConfig,
    SimulationState,
)

LAPLACIAN_KERNEL = np.array(
    [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
    dtype=np.float64,
)


class SimulationError(RuntimeError):
    """Raised when a simulation leaves the finite numerical domain."""


def laplacian(values: FloatArray, output: FloatArray | None = None) -> FloatArray:
    """Apply the measured 3×3 Gray-Scott stencil with periodic boundaries."""

    if values.ndim != 2:
        raise ValueError("laplacian input must be a two-dimensional array")
    if output is None:
        output = np.empty_like(values)
    convolve(values, LAPLACIAN_KERNEL, output=output, mode="wrap")
    return output


def endpoint_fields(config: SimulationConfig, dtype: np.dtype[Any]) -> ParameterFields:
    controls = config.controls
    # These are intentionally broadcastable vectors, not repeated full-size maps.
    return ParameterFields(
        feed=np.linspace(controls.F1, controls.F2, config.width, dtype=dtype)[None, :],
        kill=np.linspace(controls.K1, controls.K2, config.width, dtype=dtype)[None, :],
        diffusion_u=np.linspace(controls.Du1, controls.Du2, config.height, dtype=dtype)[
            :, None
        ],
        diffusion_v=np.linspace(controls.Dv1, controls.Dv2, config.height, dtype=dtype)[
            :, None
        ],
    )


def initialize_state(
    config: SimulationConfig, rng: np.random.Generator
) -> SimulationState:
    dtype = np.dtype(config.dtype)
    u = np.ones(config.shape, dtype=dtype)
    v = np.zeros(config.shape, dtype=dtype)
    u += rng.random(config.shape, dtype=dtype) - dtype.type(0.5)
    v += rng.random(config.shape, dtype=dtype) - dtype.type(0.5)
    return SimulationState(
        u=u,
        v=v,
        laplacian_u=np.empty_like(u),
        laplacian_v=np.empty_like(v),
        reaction=np.empty_like(u),
    )


def advance(
    state: SimulationState, fields: ParameterFields, steps: int = 1
) -> SimulationState:
    if steps <= 0:
        raise ValueError("steps must be positive")

    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            laplacian(state.u, state.laplacian_u)
            laplacian(state.v, state.laplacian_v)
            np.multiply(state.v, state.v, out=state.reaction)
            np.multiply(state.u, state.reaction, out=state.reaction)
            state.u += (
                fields.diffusion_u * state.laplacian_u
                - state.reaction
                + fields.feed * (1 - state.u)
            )
            state.v += (
                fields.diffusion_v * state.laplacian_v
                + state.reaction
                - (fields.feed + fields.kill) * state.v
            )
            state.iteration += 1

    if not np.isfinite(state.u).all() or not np.isfinite(state.v).all():
        raise SimulationError("simulation produced non-finite concentrations")
    return state


def normalize_image(values: FloatArray) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if not np.isfinite(values).all():
        raise SimulationError("cannot normalize non-finite concentrations")
    minimum = values.min()
    value_range = values.max() - minimum
    if value_range <= np.finfo(values.dtype).eps:
        return np.zeros_like(values, dtype=np.uint8)
    return (255 * (values - minimum) / value_range).astype(np.uint8)


class TuringSimulator:
    """Stateful live simulator built on the transport-independent engine."""

    def __init__(
        self,
        controls: ControlValues | Mapping[str, float],
        shape: tuple[int, int] = (256, 256),
        seed: int = 0,
        dtype: str = "float32",
    ):
        if not isinstance(controls, ControlValues):
            controls = ControlValues.from_mapping(controls)
        self.config = SimulationConfig(
            width=shape[1],
            height=shape[0],
            controls=controls,
            seed=seed,
            dtype=dtype,
        )
        self.dtype = np.dtype(dtype)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.fields = endpoint_fields(self.config, self.dtype)
        self.state = initialize_state(self.config, self.rng)

    @property
    def shape(self) -> tuple[int, int]:
        return self.config.shape

    @property
    def U(self) -> FloatArray:
        return self.state.u

    @property
    def V(self) -> FloatArray:
        return self.state.v

    @property
    def F(self) -> FloatArray:
        return self.fields.feed

    @property
    def k(self) -> FloatArray:
        return self.fields.kill

    @property
    def Du(self) -> FloatArray:
        return self.fields.diffusion_u

    @property
    def Dv(self) -> FloatArray:
        return self.fields.diffusion_v

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            if not 0 <= seed <= 4_294_967_295:
                raise ValueError("seed must fit in an unsigned 32-bit integer")
            self.seed = seed
            self.config = replace(self.config, seed=seed)
        self.rng = np.random.default_rng(self.seed)
        self.state = initialize_state(self.config, self.rng)

    def perturb(self, noise: float = 0.25) -> None:
        if not np.isfinite(noise) or not 0 < noise <= 1:
            raise ValueError("noise must be finite, greater than zero, and at most one")
        amplitude = self.dtype.type(noise)
        half = self.dtype.type(0.5)
        self.state.u += amplitude * (
            self.rng.random(self.shape, dtype=self.dtype) - half
        )
        self.state.v += amplitude * (
            self.rng.random(self.shape, dtype=self.dtype) - half
        )

    def update_controls(self, controls: ControlValues | Mapping[str, float]) -> None:
        if not isinstance(controls, ControlValues):
            controls = ControlValues.from_mapping(controls)
        self.config = SimulationConfig(
            width=self.config.width,
            height=self.config.height,
            controls=controls,
            seed=self.seed,
            dtype=self.config.dtype,
            boundary=self.config.boundary,
        )
        self.fields = endpoint_fields(self.config, self.dtype)

    def react(self) -> None:
        advance(self.state, self.fields)

    def step(self, steps: int = 1) -> np.ndarray[Any, np.dtype[np.uint8]]:
        advance(self.state, self.fields, steps)
        return normalize_image(self.state.v)

    img_norm = staticmethod(normalize_image)
