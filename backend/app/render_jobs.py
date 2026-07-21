from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic, time
from typing import Any
from uuid import uuid4

from PIL import Image, PngImagePlugin

from app.api.schemas import RenderPlanRequest
from app.config import Settings
from app.core.engine import SimulationError, TuringSimulator
from app.core.models import ENGINE_VERSION
from app.runtime import ComputeRuntime
from app.usage import UsageStore

ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("completed", "failed", "cancelled", "expired", "interrupted")


class RenderQueueFullError(RuntimeError):
    pass


class ClientRenderLimitError(RuntimeError):
    pass


class RenderJobNotFoundError(LookupError):
    pass


class RenderArtifactUnavailableError(LookupError):
    pass


def _iso_timestamp(value: float | None) -> str | None:
    if value is None:
        return None
    return datetime.fromtimestamp(value, UTC).isoformat()


def _render_artifact(
    frame: Any,
    payload: RenderPlanRequest,
    plan: dict[str, Any],
    destination: Path,
) -> None:
    image = Image.fromarray(frame).convert("L")
    output_width = int(plan["output_width"])
    output_height = int(plan["output_height"])

    if payload.framing == "crop" and output_width != output_height:
        target_ratio = output_width / output_height
        source_width, source_height = image.size
        source_ratio = source_width / source_height
        if source_ratio > target_ratio:
            cropped_width = round(source_height * target_ratio)
            left = (source_width - cropped_width) // 2
            image = image.crop((left, 0, left + cropped_width, source_height))
        else:
            cropped_height = round(source_width / target_ratio)
            top = (source_height - cropped_height) // 2
            image = image.crop((0, top, source_width, top + cropped_height))

    if payload.framing == "fit" and output_width != output_height:
        image.thumbnail((output_width, output_height), Image.Resampling.BICUBIC)
        canvas = Image.new("L", (output_width, output_height), color=0)
        canvas.paste(
            image,
            ((output_width - image.width) // 2, (output_height - image.height) // 2),
        )
        image = canvas
    elif image.size != (output_width, output_height):
        image = image.resize((output_width, output_height), Image.Resampling.BICUBIC)

    metadata_payload = {
        "render_version": 1,
        "engine_version": ENGINE_VERSION,
        "boundary": "periodic",
        "dtype": "float32",
        "actual_steps": payload.development_steps,
        "recipe": {
            "recipe_version": 1,
            "engine_version": ENGINE_VERSION,
            "name": payload.recipe_name,
            "preset": payload.recipe_preset,
            "controls": payload.controls.model_dump(),
            "seed": payload.seed,
        },
        "plan": plan,
    }
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text(
        "TuringParams", json.dumps(metadata_payload, separators=(",", ":"))
    )

    temporary = destination.with_suffix(".partial")
    image.save(
        temporary,
        format="PNG",
        pnginfo=metadata,
        dpi=(int(plan["pixels_per_inch"]), int(plan["pixels_per_inch"])),
    )
    os.replace(temporary, destination)


class RenderJobManager:
    def __init__(self, config: Settings, runtime: ComputeRuntime, usage: UsageStore):
        self.config = config
        self.runtime = runtime
        self.usage = usage
        self.data_dir = Path(config.render_data_dir).resolve()
        self.artifact_dir = self.data_dir / "artifacts"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        for partial in self.artifact_dir.glob("*.partial"):
            partial.unlink(missing_ok=True)
        self._database = sqlite3.connect(
            self.data_dir / "render-jobs.sqlite3", check_same_thread=False
        )
        self._database.row_factory = sqlite3.Row
        self._database.execute("PRAGMA journal_mode=WAL")
        self._database.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()
        self._wake = asyncio.Event()
        self._worker_task: asyncio.Task[None] | None = None
        self._closing = False
        self._initialize_database()
        self._trim_history()

    def _initialize_database(self) -> None:
        with self._lock, self._database:
            self._database.execute(
                """
                CREATE TABLE IF NOT EXISTS render_jobs (
                    id TEXT PRIMARY KEY,
                    client_key TEXT NOT NULL,
                    state TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    finished_at REAL,
                    progress_steps INTEGER NOT NULL DEFAULT 0,
                    requested_steps INTEGER NOT NULL,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    artifact_path TEXT,
                    expires_at REAL
                )
                """
            )
            now = time()
            cursor = self._database.execute(
                """
                UPDATE render_jobs
                SET state = 'interrupted', finished_at = ?,
                    error = 'The server restarted while this render was running.'
                WHERE state = 'running'
                """,
                (now,),
            )
            if cursor.rowcount:
                self.usage.increment("render_failed", cursor.rowcount)
            self._database.execute(
                "CREATE INDEX IF NOT EXISTS render_jobs_state_created "
                "ON render_jobs(state, created_at)"
            )

    async def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(
                self._worker(), name="render-job-worker"
            )
            self._wake.set()

    async def close(self) -> None:
        self._closing = True
        self._wake.set()
        if self._worker_task is not None:
            await self._worker_task
        with self._lock:
            self._database.close()

    def enqueue(
        self,
        payload: RenderPlanRequest,
        plan: dict[str, Any],
        client_key: str,
    ) -> dict[str, Any]:
        now = time()
        job_id = uuid4().hex
        with self._lock, self._database:
            queued = self._database.execute(
                "SELECT COUNT(*) FROM render_jobs WHERE state = 'queued'"
            ).fetchone()[0]
            if queued >= self.config.max_render_queue:
                self.runtime.metrics.increment("render_jobs_rejected")
                self.usage.increment("render_rejected")
                raise RenderQueueFullError("The render queue is full.")
            client_active = self._database.execute(
                "SELECT COUNT(*) FROM render_jobs "
                "WHERE client_key = ? AND state IN ('queued', 'running')",
                (client_key,),
            ).fetchone()[0]
            if client_active >= self.config.max_render_jobs_per_client:
                self.runtime.metrics.increment("render_jobs_rejected")
                self.usage.increment("render_rejected")
                raise ClientRenderLimitError(
                    "This client already has the maximum number of active renders."
                )
            self._database.execute(
                """
                INSERT INTO render_jobs (
                    id, client_key, state, payload_json, plan_json, created_at,
                    requested_steps
                ) VALUES (?, ?, 'queued', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    client_key,
                    payload.model_dump_json(),
                    json.dumps(plan, separators=(",", ":")),
                    now,
                    payload.development_steps,
                ),
            )
        self.runtime.metrics.increment("render_jobs_queued")
        self._wake.set()
        job = self.get(job_id)
        self._trim_history()
        return job

    def get(self, job_id: str) -> dict[str, Any]:
        self._expire_artifacts()
        self._trim_history()
        with self._lock:
            row = self._database.execute(
                "SELECT * FROM render_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        if row is None:
            raise RenderJobNotFoundError(job_id)
        return self._serialize(row)

    def cancel(self, job_id: str) -> dict[str, Any]:
        now = time()
        with self._lock, self._database:
            row = self._database.execute(
                "SELECT state FROM render_jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise RenderJobNotFoundError(job_id)
            if row["state"] == "queued":
                self._database.execute(
                    "UPDATE render_jobs SET state = 'cancelled', finished_at = ? "
                    "WHERE id = ?",
                    (now, job_id),
                )
                self.runtime.metrics.increment("render_jobs_cancelled")
                self.usage.increment("render_cancelled")
            elif row["state"] == "running":
                self._database.execute(
                    "UPDATE render_jobs SET cancel_requested = 1 WHERE id = ?",
                    (job_id,),
                )
        self._wake.set()
        return self.get(job_id)

    def artifact(self, job_id: str) -> tuple[Path, str]:
        job = self.get(job_id)
        if job["state"] != "completed" or not job["artifact_available"]:
            raise RenderArtifactUnavailableError(job_id)
        with self._lock:
            row = self._database.execute(
                "SELECT artifact_path FROM render_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        path = Path(row["artifact_path"]).resolve()
        if path.parent != self.artifact_dir or not path.is_file():
            raise RenderArtifactUnavailableError(job_id)
        plan = job["plan"]
        filename = (
            f"turing-{job_id[:10]}-{plan['output_width']}x{plan['output_height']}-"
            f"{job['requested_steps']}-steps.png"
        )
        return path, filename

    def _serialize(self, row: sqlite3.Row) -> dict[str, Any]:
        requested = int(row["requested_steps"])
        progress = int(row["progress_steps"])
        queue_position = None
        if row["state"] == "queued":
            with self._lock:
                queue_position = self._database.execute(
                    "SELECT COUNT(*) FROM render_jobs "
                    "WHERE state = 'queued' AND created_at <= ?",
                    (row["created_at"],),
                ).fetchone()[0]
        artifact_path = row["artifact_path"]
        artifact_available = bool(
            row["state"] == "completed"
            and artifact_path
            and Path(artifact_path).is_file()
        )
        return {
            "id": row["id"],
            "state": row["state"],
            "created_at": _iso_timestamp(row["created_at"]),
            "started_at": _iso_timestamp(row["started_at"]),
            "finished_at": _iso_timestamp(row["finished_at"]),
            "expires_at": _iso_timestamp(row["expires_at"]),
            "progress_steps": progress,
            "requested_steps": requested,
            "progress_percent": round(100 * progress / requested, 1),
            "queue_position": queue_position,
            "cancel_requested": bool(row["cancel_requested"]),
            "error": row["error"],
            "artifact_available": artifact_available,
            "artifact_url": (
                f"/api/v1/renders/{row['id']}/artifact" if artifact_available else None
            ),
            "plan": json.loads(row["plan_json"]),
        }

    def _next_queued(self) -> sqlite3.Row | None:
        with self._lock:
            return self._database.execute(
                "SELECT * FROM render_jobs WHERE state = 'queued' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()

    def _claim(self, job_id: str) -> bool:
        with self._lock, self._database:
            cursor = self._database.execute(
                "UPDATE render_jobs SET state = 'running', started_at = ? "
                "WHERE id = ? AND state = 'queued'",
                (time(), job_id),
            )
        return cursor.rowcount == 1

    def _update(self, job_id: str, **values: Any) -> None:
        if not values:
            return
        assignments = ", ".join(f"{name} = ?" for name in values)
        with self._lock, self._database:
            self._database.execute(
                f"UPDATE render_jobs SET {assignments} WHERE id = ?",  # noqa: S608
                (*values.values(), job_id),
            )

    def _cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._database.execute(
                "SELECT cancel_requested FROM render_jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    async def _worker(self) -> None:
        while not self._closing:
            self._expire_artifacts()
            row = self._next_queued()
            if row is None:
                self._wake.clear()
                if self._next_queued() is not None:
                    continue
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=60.0)
                except TimeoutError:
                    pass
                continue

            if not await self.runtime.gate.acquire():
                await asyncio.sleep(0.25)
                continue
            try:
                if self._closing:
                    return
                if not self._claim(row["id"]):
                    continue
                self.runtime.metrics.increment("render_jobs_started")
                self.usage.record_peak("peak_compute_active", self.runtime.gate.active)
                await self._run_job(row)
            finally:
                self.runtime.gate.release()

    async def _run_job(self, row: sqlite3.Row) -> None:
        job_id = row["id"]
        payload = RenderPlanRequest.model_validate_json(row["payload_json"])
        plan: dict[str, Any] = json.loads(row["plan_json"])
        started = monotonic()
        try:
            simulator = await self.runtime.run(
                TuringSimulator,
                payload.controls.model_dump(),
                (int(plan["simulation_height"]), int(plan["simulation_width"])),
                payload.seed,
            )
            frame = None
            while simulator.state.iteration < payload.development_steps:
                if self._closing:
                    self._update(
                        job_id,
                        state="interrupted",
                        finished_at=time(),
                        error="The server shut down before this render completed.",
                    )
                    self.runtime.metrics.increment("render_jobs_failed")
                    self.usage.increment("render_failed")
                    return
                if self._cancel_requested(job_id):
                    self._update(job_id, state="cancelled", finished_at=time())
                    self.runtime.metrics.increment("render_jobs_cancelled")
                    self.usage.increment("render_cancelled")
                    return
                if monotonic() - started > self.config.render_job_timeout_seconds:
                    self._update(
                        job_id,
                        state="failed",
                        finished_at=time(),
                        error="The render exceeded its configured execution timeout.",
                    )
                    self.runtime.metrics.increment("render_jobs_failed")
                    self.usage.increment("render_failed")
                    return
                chunk = min(
                    self.config.render_chunk_steps,
                    payload.development_steps - simulator.state.iteration,
                )
                frame = await self.runtime.run(simulator.step, chunk)
                self._update(job_id, progress_steps=simulator.state.iteration)

            if frame is None:
                raise RuntimeError("render completed without producing a frame")
            destination = self.artifact_dir / f"{job_id}.png"
            await self.runtime.run(_render_artifact, frame, payload, plan, destination)
            finished = time()
            self._update(
                job_id,
                state="completed",
                finished_at=finished,
                progress_steps=payload.development_steps,
                artifact_path=str(destination),
                expires_at=finished + self.config.render_artifact_ttl_seconds,
            )
            self.runtime.metrics.increment("render_jobs_completed")
            self.usage.increment("render_completed")
            self._trim_artifacts()
        except SimulationError as error:
            self.runtime.metrics.increment("numerical_failures")
            self.usage.increment("numerical_failures")
            self._update(
                job_id,
                state="failed",
                finished_at=time(),
                error=f"The simulation failed: {error}",
            )
            self.runtime.metrics.increment("render_jobs_failed")
            self.usage.increment("render_failed")
        except Exception as error:  # noqa: BLE001 - persisted for job diagnosis
            self._update(
                job_id,
                state="failed",
                finished_at=time(),
                error=f"The render failed: {error}",
            )
            self.runtime.metrics.increment("render_jobs_failed")
            self.usage.increment_many({"render_failed": 1, "internal_errors": 1})
        finally:
            self.usage.increment("render_seconds", monotonic() - started)
            self._trim_history()

    def _expire_artifacts(self) -> None:
        now = time()
        with self._lock, self._database:
            rows = self._database.execute(
                "SELECT id, artifact_path FROM render_jobs "
                "WHERE state = 'completed' AND expires_at <= ?",
                (now,),
            ).fetchall()
            for row in rows:
                if row["artifact_path"]:
                    Path(row["artifact_path"]).unlink(missing_ok=True)
                self._database.execute(
                    "UPDATE render_jobs SET state = 'expired', artifact_path = NULL "
                    "WHERE id = ?",
                    (row["id"],),
                )

    def _trim_artifacts(self) -> None:
        with self._lock, self._database:
            rows = self._database.execute(
                "SELECT id, artifact_path FROM render_jobs "
                "WHERE state = 'completed' ORDER BY finished_at DESC "
                "LIMIT -1 OFFSET ?",
                (self.config.max_render_artifacts,),
            ).fetchall()
            for row in rows:
                if row["artifact_path"]:
                    Path(row["artifact_path"]).unlink(missing_ok=True)
                self._database.execute(
                    "UPDATE render_jobs SET state = 'expired', artifact_path = NULL "
                    "WHERE id = ?",
                    (row["id"],),
                )

    def _trim_history(self) -> None:
        placeholders = ", ".join("?" for _ in TERMINAL_STATES)
        with self._lock, self._database:
            rows = self._database.execute(
                f"SELECT id, artifact_path FROM render_jobs "  # noqa: S608
                f"WHERE state IN ({placeholders}) "
                "ORDER BY finished_at DESC, created_at DESC "
                "LIMIT -1 OFFSET ?",
                (*TERMINAL_STATES, self.config.max_render_job_history),
            ).fetchall()
            for row in rows:
                if row["artifact_path"]:
                    Path(row["artifact_path"]).unlink(missing_ok=True)
                self._database.execute(
                    "DELETE FROM render_jobs WHERE id = ?", (row["id"],)
                )


__all__ = [
    "ClientRenderLimitError",
    "RenderArtifactUnavailableError",
    "RenderJobManager",
    "RenderJobNotFoundError",
    "RenderQueueFullError",
    "TERMINAL_STATES",
]
