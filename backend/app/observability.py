from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime

import psutil

LOG_FIELDS = (
    "event",
    "request_id",
    "session_id",
    "duration_seconds",
    "frame_bytes",
    "error_code",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        for field_name in LOG_FIELDS:
            value = getattr(record, field_name, None)
            if value is not None:
                payload[field_name] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, separators=(",", ":"))


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


@dataclass(slots=True)
class Metrics:
    sessions_started: int = 0
    sessions_finished: int = 0
    sessions_rejected: int = 0
    renders_started: int = 0
    renders_finished: int = 0
    renders_rejected: int = 0
    numerical_failures: int = 0
    frames: int = 0
    frame_step_seconds: float = 0.0
    frame_encode_seconds: float = 0.0
    frame_bytes: int = 0
    render_seconds: float = 0.0
    render_simulation_seconds: float = 0.0
    render_encode_seconds: float = 0.0
    event_loop_lag_seconds: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def increment(self, field_name: str) -> None:
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + 1)

    def observe_frame(
        self, step_seconds: float, encode_seconds: float, frame_bytes: int
    ) -> None:
        with self._lock:
            self.frames += 1
            self.frame_step_seconds += step_seconds
            self.frame_encode_seconds += encode_seconds
            self.frame_bytes += frame_bytes

    def observe_render(
        self,
        duration_seconds: float,
        simulation_seconds: float,
        encode_seconds: float,
    ) -> None:
        with self._lock:
            self.render_seconds += duration_seconds
            self.render_simulation_seconds += simulation_seconds
            self.render_encode_seconds += encode_seconds

    def set_event_loop_lag(self, seconds: float) -> None:
        with self._lock:
            self.event_loop_lag_seconds = max(0.0, seconds)

    def prometheus_text(self, *, active: int, waiting: int, capacity: int) -> str:
        with self._lock:
            values = {
                "turing_compute_active": active,
                "turing_compute_waiting": waiting,
                "turing_compute_capacity": capacity,
                "turing_sessions_started_total": self.sessions_started,
                "turing_sessions_finished_total": self.sessions_finished,
                "turing_sessions_rejected_total": self.sessions_rejected,
                "turing_renders_started_total": self.renders_started,
                "turing_renders_finished_total": self.renders_finished,
                "turing_renders_rejected_total": self.renders_rejected,
                "turing_numerical_failures_total": self.numerical_failures,
                "turing_frames_total": self.frames,
                "turing_frame_step_seconds_total": self.frame_step_seconds,
                "turing_frame_encode_seconds_total": self.frame_encode_seconds,
                "turing_frame_bytes_total": self.frame_bytes,
                "turing_render_seconds_total": self.render_seconds,
                "turing_render_simulation_seconds_total": (
                    self.render_simulation_seconds
                ),
                "turing_render_encode_seconds_total": self.render_encode_seconds,
                "turing_event_loop_lag_seconds": self.event_loop_lag_seconds,
                "turing_process_resident_memory_bytes": psutil.Process()
                .memory_info()
                .rss,
            }
        return "\n".join(f"{name} {value}" for name, value in values.items()) + "\n"
