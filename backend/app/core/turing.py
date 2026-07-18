"""Batch compatibility layer around the transport-independent simulation engine."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, PngImagePlugin
from scipy.ndimage import zoom

from app.core.engine import (
    SimulationError,
    TuringSimulator,
    advance,
    initialize_state,
    laplacian,
    normalize_image,
)
from app.core.models import ENGINE_VERSION, ParameterFields, SimulationConfig

Axis = Literal["x", "y"]
ProgressCallback = Callable[[int, int], None]


def parameter_map(
    w: int,
    h: int,
    x_ctrl: Sequence[float],
    p_vals: Sequence[float],
    axis: Axis = "x",
    *,
    dtype: str = "float32",
) -> np.ndarray[Any, np.dtype[Any]]:
    """Return a broadcastable interpolated parameter vector."""

    if w <= 0 or h <= 0:
        raise ValueError("w and h must be integers greater than zero")
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")
    if not x_ctrl or not p_vals or len(x_ctrl) != len(p_vals):
        raise ValueError(
            "control positions and values must be non-empty and equal length"
        )

    positions = np.asarray(x_ctrl, dtype=dtype)
    values = np.asarray(p_vals, dtype=dtype)
    if not np.isfinite(positions).all() or not np.isfinite(values).all():
        raise ValueError("control positions and values must be finite")
    if np.any(np.diff(positions) <= 0) or positions[0] != 0 or positions[-1] != 1:
        raise ValueError("control positions must increase strictly from 0 to 1")

    samples = np.linspace(0, 1, w if axis == "x" else h, dtype=dtype)
    interpolated = np.interp(samples, positions, values).astype(dtype, copy=False)
    return interpolated[None, :] if axis == "x" else interpolated[:, None]


def turing_pattern(
    w: int = 512,
    h: int = 128,
    Du_ctrl: Sequence[float] = (0.0, 1.0),
    Du_vals: Sequence[float] = (0.7, 0.7),
    Du_axis: Axis = "y",
    Dv_ctrl: Sequence[float] = (0.0, 1.0),
    Dv_vals: Sequence[float] = (0.25, 0.25),
    Dv_axis: Axis = "y",
    F_ctrl: Sequence[float] = (0.0, 1.0),
    F_vals: Sequence[float] = (0.04, 0.08),
    F_axis: Axis = "x",
    k_ctrl: Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    k_vals: Sequence[float] = (0.056, 0.06, 0.0635, 0.0665, 0.07, 0.074),
    k_axis: Axis = "x",
    steps: int = 10_000,
    upsample: int = 2,
    seed: int = 0,
    dtype: str = "float32",
    progress: ProgressCallback | None = None,
) -> np.ndarray[Any, np.dtype[np.uint8]]:
    if steps <= 0 or upsample <= 0:
        raise ValueError("steps and upsample must be positive")

    # The controls object is unused for arbitrary maps, but SimulationConfig owns
    # shape/dtype/seed validation for both batch and live modes.
    from app.core.models import ControlValues

    config = SimulationConfig(
        width=w,
        height=h,
        controls=ControlValues(0.04, 0.04, 0.06, 0.06, 0.7, 0.7, 0.25, 0.25),
        seed=seed,
        dtype=dtype,
    )
    rng = np.random.default_rng(seed)
    state = initialize_state(config, rng)
    fields = ParameterFields(
        diffusion_u=parameter_map(w, h, Du_ctrl, Du_vals, Du_axis, dtype=dtype),
        diffusion_v=parameter_map(w, h, Dv_ctrl, Dv_vals, Dv_axis, dtype=dtype),
        feed=parameter_map(w, h, F_ctrl, F_vals, F_axis, dtype=dtype),
        kill=parameter_map(w, h, k_ctrl, k_vals, k_axis, dtype=dtype),
    )

    for iteration in range(steps):
        advance(state, fields)
        if iteration % 1000 == 0 and iteration <= steps // 2:
            state.v += np.asarray(0.5, dtype=dtype) * (
                rng.random(config.shape, dtype=np.dtype(dtype))
                - np.asarray(0.5, dtype=dtype)
            )
        if iteration % 250 == 0:
            value_range = state.v.max() - state.v.min()
            if value_range < 1e-5:
                raise SimulationError(
                    f"model collapsed by step {iteration}; range={value_range:g}"
                )
            if progress is not None:
                progress(iteration, steps)

    image = normalize_image(state.v)
    if upsample == 1:
        return image
    return zoom(image, upsample, order=3)


def main() -> None:
    module_directory = Path(__file__).resolve().parent
    repository_root = module_directory.parents[2]
    parameters = json.loads((module_directory / "turing_parameters.json").read_text())
    parameters.setdefault("seed", 0)
    pattern = turing_pattern(**parameters)

    metadata_payload = {
        "engine_version": ENGINE_VERSION,
        "boundary": "periodic",
        **parameters,
    }
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("TuringParams", json.dumps(metadata_payload, indent=2))

    output_directory = repository_root / "images"
    output_directory.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    Image.fromarray(pattern).convert("L").save(
        output_directory / f"turing_pattern_{timestamp}.png", pnginfo=metadata
    )


__all__ = [
    "SimulationError",
    "TuringSimulator",
    "laplacian",
    "parameter_map",
    "turing_pattern",
]


if __name__ == "__main__":
    main()
