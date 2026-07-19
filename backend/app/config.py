from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass


def _get_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    value = default if raw is None else int(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _get_float(name: str, default: float, *, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    value = default if raw is None else float(raw)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _get_origins(name: str) -> tuple[str, ...]:
    raw = os.getenv(name, "http://localhost:3000,http://localhost:5173")
    origins = tuple(
        origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()
    )
    if not origins:
        raise ValueError(f"{name} must contain at least one origin")
    if "*" in origins:
        raise ValueError(f"{name} may not contain a wildcard origin")
    return origins


def _get_log_level(name: str, default: str = "INFO") -> str:
    value = os.getenv(name, default).upper()
    if value not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError(f"{name} must be a standard Python log level")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    allowed_origins: tuple[str, ...]
    allow_originless_websockets: bool
    docs_enabled: bool
    max_compute_jobs: int
    max_compute_waiters: int
    compute_workers: int
    admission_timeout_seconds: float
    initial_message_timeout_seconds: float
    idle_timeout_seconds: float
    frame_rate: float
    steps_per_frame: int
    preview_size: int
    max_websocket_message_bytes: int
    render_size: int
    render_steps: int
    render_upsample: int
    max_render_simulation_pixels: int
    max_render_output_edge: int
    benchmark_iterations_per_second: float
    render_data_dir: str
    max_render_queue: int
    max_render_jobs_per_client: int
    render_job_timeout_seconds: float
    render_artifact_ttl_seconds: int
    max_render_artifacts: int
    max_render_job_history: int
    render_chunk_steps: int
    log_level: str

    @classmethod
    def from_env(cls) -> Settings:
        max_jobs = _get_int("TURING_MAX_COMPUTE_JOBS", 2, minimum=1, maximum=16)
        workers = _get_int("TURING_COMPUTE_WORKERS", 2, minimum=1, maximum=16)
        if workers > max_jobs:
            raise ValueError(
                "TURING_COMPUTE_WORKERS may not exceed TURING_MAX_COMPUTE_JOBS"
            )

        return cls(
            allowed_origins=_get_origins("TURING_ALLOWED_ORIGINS"),
            allow_originless_websockets=_get_bool(
                "TURING_ALLOW_ORIGINLESS_WEBSOCKETS", True
            ),
            docs_enabled=_get_bool("TURING_ENABLE_DOCS", True),
            max_compute_jobs=max_jobs,
            max_compute_waiters=_get_int(
                "TURING_MAX_COMPUTE_WAITERS", 2, minimum=0, maximum=32
            ),
            compute_workers=workers,
            admission_timeout_seconds=_get_float(
                "TURING_ADMISSION_TIMEOUT_SECONDS", 0.25, minimum=0.01, maximum=10.0
            ),
            initial_message_timeout_seconds=_get_float(
                "TURING_INITIAL_MESSAGE_TIMEOUT_SECONDS",
                5.0,
                minimum=1.0,
                maximum=60.0,
            ),
            idle_timeout_seconds=_get_float(
                "TURING_IDLE_TIMEOUT_SECONDS", 600.0, minimum=10.0, maximum=3600.0
            ),
            frame_rate=_get_float("TURING_FRAME_RATE", 10.0, minimum=1.0, maximum=30.0),
            steps_per_frame=_get_int(
                "TURING_STEPS_PER_FRAME", 25, minimum=1, maximum=200
            ),
            preview_size=_get_int("TURING_PREVIEW_SIZE", 256, minimum=32, maximum=512),
            max_websocket_message_bytes=_get_int(
                "TURING_MAX_WEBSOCKET_MESSAGE_BYTES",
                4096,
                minimum=256,
                maximum=65_536,
            ),
            render_size=_get_int("TURING_RENDER_SIZE", 256, minimum=32, maximum=512),
            render_steps=_get_int(
                "TURING_RENDER_STEPS", 5000, minimum=100, maximum=20_000
            ),
            render_upsample=_get_int("TURING_RENDER_UPSAMPLE", 2, minimum=1, maximum=4),
            max_render_simulation_pixels=_get_int(
                "TURING_MAX_RENDER_SIMULATION_PIXELS",
                1_048_576,
                minimum=65_536,
                maximum=16_777_216,
            ),
            max_render_output_edge=_get_int(
                "TURING_MAX_RENDER_OUTPUT_EDGE", 4096, minimum=512, maximum=16_384
            ),
            benchmark_iterations_per_second=_get_float(
                "TURING_BENCHMARK_ITERATIONS_PER_SECOND",
                421.2,
                minimum=1.0,
                maximum=100_000.0,
            ),
            render_data_dir=os.getenv(
                "TURING_RENDER_DATA_DIR",
                os.path.join(tempfile.gettempdir(), "turing-pattern-renders"),
            ),
            max_render_queue=_get_int(
                "TURING_MAX_RENDER_QUEUE", 3, minimum=1, maximum=32
            ),
            max_render_jobs_per_client=_get_int(
                "TURING_MAX_RENDER_JOBS_PER_CLIENT", 2, minimum=1, maximum=8
            ),
            render_job_timeout_seconds=_get_float(
                "TURING_RENDER_JOB_TIMEOUT_SECONDS",
                900.0,
                minimum=10.0,
                maximum=7200.0,
            ),
            render_artifact_ttl_seconds=_get_int(
                "TURING_RENDER_ARTIFACT_TTL_SECONDS",
                86_400,
                minimum=300,
                maximum=604_800,
            ),
            max_render_artifacts=_get_int(
                "TURING_MAX_RENDER_ARTIFACTS", 8, minimum=1, maximum=64
            ),
            max_render_job_history=_get_int(
                "TURING_MAX_RENDER_JOB_HISTORY", 64, minimum=8, maximum=512
            ),
            render_chunk_steps=_get_int(
                "TURING_RENDER_CHUNK_STEPS", 100, minimum=10, maximum=1000
            ),
            log_level=_get_log_level("TURING_LOG_LEVEL"),
        )


settings = Settings.from_env()
