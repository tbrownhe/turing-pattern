"""Measure the numerical and frame-encoding path without network overhead."""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
from PIL import Image
from scipy.ndimage import convolve

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.turing import TuringSimulator, laplacian  # noqa: E402

CONTROLS = {
    "F1": 0.04,
    "F2": 0.08,
    "K1": 0.056,
    "K2": 0.074,
    "Du1": 0.7,
    "Du2": 0.7,
    "Dv1": 0.25,
    "Dv2": 0.25,
}
KERNEL = np.array(
    [[0.05, 0.2, 0.05], [0.2, -1.0, 0.2], [0.05, 0.2, 0.05]],
    dtype=np.float64,
)


def padded_laplacian(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="wrap")
    return (
        -values
        + 0.2
        * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
        + 0.05 * (padded[:-2, :-2] + padded[:-2, 2:] + padded[2:, :-2] + padded[2:, 2:])
    )


def roll_reference(values: np.ndarray) -> np.ndarray:
    return (
        -values
        + 0.2
        * (
            np.roll(values, 1, 0)
            + np.roll(values, -1, 0)
            + np.roll(values, 1, 1)
            + np.roll(values, -1, 1)
        )
        + 0.05
        * (
            np.roll(np.roll(values, 1, 0), 1, 1)
            + np.roll(np.roll(values, -1, 0), 1, 1)
            + np.roll(np.roll(values, 1, 0), -1, 1)
            + np.roll(np.roll(values, -1, 0), -1, 1)
        )
    )


def scipy_laplacian(values: np.ndarray) -> np.ndarray:
    return convolve(values, KERNEL, mode="wrap")


def time_function(function, values: np.ndarray, repetitions: int) -> float:
    function(values)
    started = perf_counter()
    for _ in range(repetitions):
        function(values)
    return (perf_counter() - started) / repetitions


def benchmark(size: int, steps: int, label: str) -> dict[str, object]:
    simulator = TuringSimulator(CONTROLS, shape=(size, size), seed=7)
    simulator.step(5)
    started = perf_counter()
    frame = simulator.step(steps)
    step_seconds = perf_counter() - started

    started = perf_counter()
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="PNG", optimize=False)
    encode_seconds = perf_counter() - started

    sample = np.random.default_rng(7).random((size, size))
    reference = roll_reference(sample)
    current = laplacian(sample)
    padded = padded_laplacian(sample)
    scipy_result = scipy_laplacian(sample)

    parameter_bytes = sum(
        value.nbytes for value in (simulator.F, simulator.k, simulator.Du, simulator.Dv)
    )
    state_bytes = simulator.U.nbytes + simulator.V.nbytes
    process = psutil.Process()

    return {
        "label": label,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "logical_cpus": os.cpu_count(),
        "process_threads": process.num_threads(),
        "thread_environment": {
            name: os.getenv(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
            if os.getenv(name) is not None
        },
        "numpy": np.__version__,
        "blas": np.__config__.CONFIG.get("Build Dependencies", {})
        .get("blas", {})
        .get("name", "unknown"),
        "grid": [size, size],
        "dtype": str(simulator.U.dtype),
        "steps": steps,
        "step_seconds": round(step_seconds, 6),
        "iterations_per_second": round(steps / step_seconds, 2),
        "encode_seconds": round(encode_seconds, 6),
        "frame_bytes": len(output.getvalue()),
        "state_bytes": state_bytes,
        "state_working_bytes": simulator.state.working_nbytes,
        "parameter_bytes": parameter_bytes,
        "rss_bytes": process.memory_info().rss,
        "laplacian_microseconds": {
            "roll_reference": round(time_function(roll_reference, sample, 50) * 1e6, 2),
            "current_engine": round(time_function(laplacian, sample, 50) * 1e6, 2),
            "padded_slices": round(
                time_function(padded_laplacian, sample, 50) * 1e6, 2
            ),
            "scipy_convolve": round(
                time_function(scipy_laplacian, sample, 50) * 1e6, 2
            ),
        },
        "candidate_max_absolute_difference": {
            "current_engine": float(np.max(np.abs(reference - current))),
            "padded_slices": float(np.max(np.abs(reference - padded))),
            "scipy_convolve": float(np.max(np.abs(reference - scipy_result))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--label", default="local")
    args = parser.parse_args()
    print(json.dumps(benchmark(args.size, args.steps, args.label), indent=2))


if __name__ == "__main__":
    main()
