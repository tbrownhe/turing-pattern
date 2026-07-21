from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from time import monotonic, perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from PIL import Image, PngImagePlugin
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from app import APP_VERSION
from app.api.schemas import (
    ClientMessage,
    ControlsMessage,
    PauseMessage,
    PerturbMessage,
    RenderPlanRequest,
    RenderRequest,
    ResetMessage,
    ResumeMessage,
    StartMessage,
    StepMessage,
    TimeStudyRequest,
    parse_client_message,
)
from app.config import Settings, settings
from app.core.engine import SimulationError, TuringSimulator
from app.core.models import ENGINE_VERSION
from app.observability import configure_logging
from app.render_jobs import (
    ClientRenderLimitError,
    RenderArtifactUnavailableError,
    RenderJobManager,
    RenderJobNotFoundError,
    RenderQueueFullError,
)
from app.runtime import ComputeRuntime
from app.usage import UsageStore

logger = logging.getLogger(__name__)


class ClientProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class FrameResult:
    content: bytes
    step_seconds: float
    encode_seconds: float
    iteration: int


@dataclass(frozen=True, slots=True)
class RenderResult:
    content: bytes
    simulation_seconds: float
    encode_seconds: float


QUALITY_PPI = {"draft": 150, "studio": 300, "fine": 600}
QUALITY_LABELS = {"draft": "Draft", "studio": "Studio", "fine": "Fine"}


def _encode_preview(simulator: TuringSimulator, steps: int) -> FrameResult:
    started = perf_counter()
    frame = simulator.step(steps=steps)
    step_seconds = perf_counter() - started
    started = perf_counter()
    output = BytesIO()
    Image.fromarray(frame).save(output, format="PNG", optimize=False)
    return FrameResult(
        content=output.getvalue(),
        step_seconds=step_seconds,
        encode_seconds=perf_counter() - started,
        iteration=simulator.state.iteration,
    )


def _time_study(payload: TimeStudyRequest, size: int) -> list[dict[str, int | str]]:
    simulator = TuringSimulator(
        payload.controls.model_dump(),
        shape=(size, size),
        seed=payload.seed,
    )
    checkpoints: list[dict[str, int | str]] = []
    for target in payload.checkpoints:
        frame = simulator.step(steps=target - simulator.state.iteration)
        output = BytesIO()
        Image.fromarray(frame).save(output, format="PNG", optimize=False)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        checkpoints.append(
            {
                "steps": target,
                "image_url": f"data:image/png;base64,{encoded}",
            }
        )
    return checkpoints


def _render_plan(payload: RenderPlanRequest, config: Settings) -> dict[str, object]:
    ppi = QUALITY_PPI[payload.quality]
    width_inches = payload.width if payload.unit == "in" else payload.width / 2.54
    height_inches = payload.height if payload.unit == "in" else payload.height / 2.54
    output_width = max(1, round(width_inches * ppi))
    output_height = max(1, round(height_inches * ppi))
    target_simulation_width = ceil(output_width / config.render_upsample)
    target_simulation_height = ceil(output_height / config.render_upsample)
    if payload.framing == "crop":
        simulation_width = simulation_height = max(
            target_simulation_width, target_simulation_height
        )
    elif payload.framing == "fit":
        simulation_width = simulation_height = min(
            target_simulation_width, target_simulation_height
        )
    else:
        simulation_width = target_simulation_width
        simulation_height = target_simulation_height
    simulation_pixels = simulation_width * simulation_height

    issues: list[str] = []
    if max(output_width, output_height) > config.max_render_output_edge:
        issues.append(
            "The output edge exceeds the configured "
            f"{config.max_render_output_edge:,}-pixel limit."
        )
    if simulation_pixels > config.max_render_simulation_pixels:
        issues.append(
            "The numerical grid exceeds the configured "
            f"{config.max_render_simulation_pixels:,}-cell limit."
        )
    if payload.feature_scale != 1.0:
        issues.append(
            "Fine and Bold feature scales remain unavailable until their numerical "
            "mapping is calibrated. Use Original 1x to queue this render."
        )

    preview_pixels = 256 * 256
    estimated_seconds = (
        payload.development_steps
        * simulation_pixels
        / (config.benchmark_iterations_per_second * preview_pixels)
    )
    estimated_memory_bytes = 65_000_000 + simulation_pixels * 32
    if simulation_pixels <= 262_144:
        resource_class = "light"
    elif simulation_pixels <= 589_824:
        resource_class = "moderate"
    else:
        resource_class = "heavy"

    return {
        "accepted": not issues,
        "issues": issues,
        "unit": payload.unit,
        "physical_width": payload.width,
        "physical_height": payload.height,
        "quality": payload.quality,
        "quality_label": QUALITY_LABELS[payload.quality],
        "pixels_per_inch": ppi,
        "feature_scale": payload.feature_scale,
        "scale_model_status": (
            "reference-validated"
            if payload.feature_scale == 1.0
            else "calibration-required"
        ),
        "development_steps": payload.development_steps,
        "framing": payload.framing,
        "output_width": output_width,
        "output_height": output_height,
        "simulation_width": simulation_width,
        "simulation_height": simulation_height,
        "simulation_pixels": simulation_pixels,
        "bicubic_upsample": config.render_upsample,
        "estimated_seconds_low": max(1, round(estimated_seconds * 0.85)),
        "estimated_seconds_high": max(1, round(estimated_seconds * 1.5)),
        "estimated_memory_bytes": estimated_memory_bytes,
        "resource_class": resource_class,
        "engine_version": ENGINE_VERSION,
    }


def _render_png(payload: RenderRequest, config: Settings) -> RenderResult:
    started = perf_counter()
    simulator = TuringSimulator(
        payload.controls.model_dump(),
        shape=(config.render_size, config.render_size),
        seed=payload.seed,
    )
    frame = simulator.step(steps=config.render_steps)
    simulation_seconds = perf_counter() - started
    started = perf_counter()
    image = Image.fromarray(frame).convert("L")
    if config.render_upsample != 1:
        output_size = config.render_size * config.render_upsample
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)

    metadata = PngImagePlugin.PngInfo()
    recipe = {
        "engine_version": ENGINE_VERSION,
        "boundary": "periodic",
        "dtype": simulator.config.dtype,
        "simulation_size": config.render_size,
        "steps": config.render_steps,
        "upsample": config.render_upsample,
        **payload.model_dump(),
    }
    metadata.add_text(
        "TuringParams",
        json.dumps(recipe, indent=2),
    )
    output = BytesIO()
    image.save(output, format="PNG", pnginfo=metadata)
    return RenderResult(
        content=output.getvalue(),
        simulation_seconds=simulation_seconds,
        encode_seconds=perf_counter() - started,
    )


async def _receive_message(websocket: WebSocket, config: Settings) -> ClientMessage:
    try:
        raw = await websocket.receive_text()
    except WebSocketDisconnect:
        raise
    except RuntimeError as error:
        raise ClientProtocolError(
            "text_message_required", "Client messages must be JSON text."
        ) from error

    if len(raw.encode("utf-8")) > config.max_websocket_message_bytes:
        raise ClientProtocolError(
            "message_too_large",
            f"Messages may not exceed {config.max_websocket_message_bytes} bytes.",
        )

    try:
        return parse_client_message(raw)
    except ValidationError as error:
        raise ClientProtocolError(
            "invalid_message", "Message does not match protocol version 1."
        ) from error


async def _send_error_and_close(
    websocket: WebSocket, error_code: str, message: str, close_code: int
) -> None:
    with suppress(RuntimeError, WebSocketDisconnect):
        await websocket.send_json(
            {"type": "error", "error": {"code": error_code, "message": message}}
        )
    with suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=close_code, reason=message[:120])


def _origin_is_allowed(websocket: WebSocket, config: Settings) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return config.allow_originless_websockets
    return origin.rstrip("/") in config.allowed_origins


def create_app(config: Settings = settings) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(config.log_level)
        usage = UsageStore(config.render_data_dir, config.report_timezone)
        usage.increment("backend_starts")
        application.state.usage = usage
        runtime = ComputeRuntime(config)
        application.state.runtime = runtime
        render_jobs = RenderJobManager(config, runtime, usage)
        application.state.render_jobs = render_jobs
        await render_jobs.start()
        monitor_stopped = asyncio.Event()

        async def monitor_event_loop() -> None:
            interval = 1.0
            expected = monotonic() + interval
            while not monitor_stopped.is_set():
                await asyncio.sleep(interval)
                now = monotonic()
                runtime.metrics.set_event_loop_lag(now - expected)
                expected = now + interval

        monitor_task = asyncio.create_task(
            monitor_event_loop(), name="event-loop-monitor"
        )
        try:
            yield
        finally:
            monitor_stopped.set()
            monitor_task.cancel()
            await asyncio.gather(monitor_task, return_exceptions=True)
            await render_jobs.close()
            runtime.close()
            usage.close()

    application = FastAPI(
        title="Turing Pattern API",
        version=APP_VERSION,
        docs_url="/docs" if config.docs_enabled else None,
        redoc_url="/redoc" if config.docs_enabled else None,
        openapi_url="/openapi.json" if config.docs_enabled else None,
        lifespan=lifespan,
    )
    application.state.settings = config
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type"],
    )

    @application.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/readyz", include_in_schema=False)
    async def readiness(request: Request) -> dict[str, int | str]:
        runtime: ComputeRuntime = request.app.state.runtime
        return {
            "status": "ready",
            "active_compute_jobs": runtime.gate.active,
            "waiting_compute_jobs": runtime.gate.waiting,
            "compute_capacity": runtime.gate.capacity,
        }

    @application.get(
        "/metrics", response_class=PlainTextResponse, include_in_schema=False
    )
    async def metrics(request: Request) -> PlainTextResponse:
        runtime: ComputeRuntime = request.app.state.runtime
        return PlainTextResponse(
            runtime.metrics.prometheus_text(
                active=runtime.gate.active,
                waiting=runtime.gate.waiting,
                capacity=runtime.gate.capacity,
            ),
            media_type="text/plain; version=0.0.4",
        )

    @application.post("/api/v1/render-plans")
    async def plan_render(payload: RenderPlanRequest) -> dict[str, object]:
        return _render_plan(payload, config)

    def get_render_job_or_404(
        manager: RenderJobManager, job_id: str
    ) -> dict[str, object]:
        try:
            return manager.get(job_id)
        except RenderJobNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "render_not_found", "message": "Render job not found."},
            ) from error

    @application.post("/api/v1/renders", status_code=202)
    async def queue_render(
        payload: RenderPlanRequest, request: Request, response: Response
    ) -> dict[str, object]:
        usage: UsageStore = request.app.state.usage
        usage.increment("render_requests")
        plan = _render_plan(payload, config)
        if not plan["accepted"]:
            usage.increment("render_rejected")
            raise HTTPException(
                status_code=422,
                detail={
                    "code": "render_plan_rejected",
                    "message": "The render plan is not executable.",
                    "issues": plan["issues"],
                },
            )
        manager: RenderJobManager = request.app.state.render_jobs
        forwarded = request.headers.get("x-real-ip")
        client_key = forwarded or (request.client.host if request.client else "unknown")
        try:
            job = manager.enqueue(payload, plan, client_key)
        except ClientRenderLimitError as error:
            raise HTTPException(
                status_code=429,
                detail={"code": "client_render_limit", "message": str(error)},
                headers={"Retry-After": "5"},
            ) from error
        except RenderQueueFullError as error:
            raise HTTPException(
                status_code=503,
                detail={"code": "render_queue_full", "message": str(error)},
                headers={"Retry-After": "5"},
            ) from error
        response.headers["Location"] = f"/api/v1/renders/{job['id']}"
        response.headers["Cache-Control"] = "no-store"
        return job

    @application.get("/api/v1/renders/{job_id}")
    async def render_status(job_id: str, request: Request) -> dict[str, object]:
        manager: RenderJobManager = request.app.state.render_jobs
        return get_render_job_or_404(manager, job_id)

    @application.delete("/api/v1/renders/{job_id}", status_code=202)
    async def cancel_render(job_id: str, request: Request) -> dict[str, object]:
        manager: RenderJobManager = request.app.state.render_jobs
        try:
            return manager.cancel(job_id)
        except RenderJobNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "render_not_found", "message": "Render job not found."},
            ) from error

    @application.get("/api/v1/renders/{job_id}/artifact")
    async def render_artifact(job_id: str, request: Request) -> FileResponse:
        manager: RenderJobManager = request.app.state.render_jobs
        try:
            path, filename = manager.artifact(job_id)
        except RenderJobNotFoundError as error:
            raise HTTPException(
                status_code=404,
                detail={"code": "render_not_found", "message": "Render job not found."},
            ) from error
        except RenderArtifactUnavailableError as error:
            raise HTTPException(
                status_code=409,
                detail={
                    "code": "artifact_unavailable",
                    "message": "The render artifact is not available.",
                },
            ) from error
        return FileResponse(
            path,
            media_type="image/png",
            filename=filename,
            headers={"Cache-Control": "private, no-store"},
        )

    @application.post("/api/v1/time-studies")
    async def create_time_study(
        payload: TimeStudyRequest, request: Request, response: Response
    ) -> dict[str, object]:
        runtime: ComputeRuntime = request.app.state.runtime
        usage: UsageStore = request.app.state.usage
        request_id = uuid4().hex
        if not await runtime.gate.acquire():
            runtime.metrics.increment("time_studies_rejected")
            usage.increment("time_studies_rejected")
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "server_busy",
                    "message": (
                        "Time-study capacity is currently full. Please retry shortly."
                    ),
                },
                headers={"Retry-After": "2", "X-Request-ID": request_id},
            )

        runtime.metrics.increment("time_studies_started")
        usage.increment("time_studies_started")
        usage.record_peak("peak_compute_active", runtime.gate.active)
        logger.info(
            "time study started",
            extra={"event": "time_study_started", "request_id": request_id},
        )
        started = monotonic()
        try:
            checkpoints = await runtime.run(_time_study, payload, config.preview_size)
        except SimulationError as error:
            runtime.metrics.increment("numerical_failures")
            usage.increment("numerical_failures")
            raise HTTPException(
                status_code=422,
                detail={"code": "simulation_failed", "message": str(error)},
                headers={"X-Request-ID": request_id},
            ) from error
        except Exception:
            usage.increment("internal_errors")
            raise
        finally:
            runtime.gate.release()

        runtime.metrics.increment("time_studies_finished")
        usage.increment("time_studies_finished")
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "time study completed",
            extra={
                "event": "time_study_completed",
                "request_id": request_id,
                "duration_seconds": monotonic() - started,
            },
        )
        return {
            "engine_version": ENGINE_VERSION,
            "simulation_size": config.preview_size,
            "seed": payload.seed,
            "checkpoints": checkpoints,
        }

    @application.post("/api/v1/generate", response_class=Response)
    async def generate(payload: RenderRequest, request: Request) -> Response:
        runtime: ComputeRuntime = request.app.state.runtime
        usage: UsageStore = request.app.state.usage
        request_id = uuid4().hex
        if not await runtime.gate.acquire():
            runtime.metrics.increment("renders_rejected")
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "server_busy",
                    "message": (
                        "Render capacity is currently full. Please retry shortly."
                    ),
                },
                headers={"Retry-After": "2", "X-Request-ID": request_id},
            )

        runtime.metrics.increment("renders_started")
        usage.record_peak("peak_compute_active", runtime.gate.active)
        logger.info(
            "render started",
            extra={"event": "render_started", "request_id": request_id},
        )
        started = monotonic()
        try:
            result = await runtime.run(_render_png, payload, config)
        except SimulationError as error:
            runtime.metrics.increment("numerical_failures")
            usage.increment("numerical_failures")
            raise HTTPException(
                status_code=422,
                detail={"code": "simulation_failed", "message": str(error)},
                headers={"X-Request-ID": request_id},
            ) from error
        except Exception:
            usage.increment("internal_errors")
            raise
        finally:
            runtime.gate.release()

        duration = monotonic() - started
        runtime.metrics.increment("renders_finished")
        runtime.metrics.observe_render(
            duration,
            result.simulation_seconds,
            result.encode_seconds,
        )
        logger.info(
            "render completed",
            extra={
                "event": "render_completed",
                "request_id": request_id,
                "duration_seconds": duration,
                "frame_bytes": len(result.content),
            },
        )
        return Response(
            content=result.content,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'attachment; filename="turing-pattern.png"',
                "X-Request-ID": request_id,
                "X-Turing-Engine": ENGINE_VERSION,
            },
        )

    @application.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        runtime: ComputeRuntime = websocket.app.state.runtime
        usage: UsageStore = websocket.app.state.usage
        session_id = uuid4().hex
        accepted = False
        tasks: set[asyncio.Task[None]] = set()
        usage_started = False
        usage_last_flush = monotonic()
        usage_frames = 0
        usage_bytes = 0

        def flush_session_usage() -> None:
            nonlocal usage_last_flush, usage_frames, usage_bytes
            if not usage_started:
                return
            now = monotonic()
            usage.increment_many(
                {
                    "live_session_seconds": now - usage_last_flush,
                    "live_frames": usage_frames,
                    "live_frame_bytes": usage_bytes,
                }
            )
            usage_last_flush = now
            usage_frames = 0
            usage_bytes = 0

        if not _origin_is_allowed(websocket, config):
            runtime.metrics.increment("sessions_rejected")
            usage.increment("live_sessions_rejected")
            logger.warning(
                "websocket origin rejected",
                extra={
                    "event": "websocket_origin_rejected",
                    "session_id": session_id,
                    "origin": websocket.headers.get("origin"),
                    "host": websocket.headers.get("host"),
                    "forwarded_proto": websocket.headers.get("x-forwarded-proto"),
                },
            )
            await websocket.close(code=1008, reason="WebSocket origin is not allowed.")
            return

        if not await runtime.gate.acquire():
            runtime.metrics.increment("sessions_rejected")
            usage.increment("live_sessions_rejected")
            await websocket.accept()
            await _send_error_and_close(
                websocket,
                "server_busy",
                "All simulation slots are busy. Please retry shortly.",
                1013,
            )
            return

        runtime.metrics.increment("sessions_started")
        usage.increment("live_sessions_admitted")
        usage.record_peak("peak_compute_active", runtime.gate.active)
        usage_started = True
        usage_last_flush = monotonic()
        logger.info(
            "websocket started",
            extra={"event": "websocket_started", "session_id": session_id},
        )
        started = monotonic()
        try:
            await websocket.accept()
            accepted = True
            try:
                initial = await asyncio.wait_for(
                    _receive_message(websocket, config),
                    timeout=config.initial_message_timeout_seconds,
                )
            except TimeoutError as error:
                raise ClientProtocolError(
                    "initial_message_timeout",
                    "The simulation start message was not received in time.",
                ) from error

            if not isinstance(initial, StartMessage):
                raise ClientProtocolError(
                    "start_required", "The first message must have type 'start'."
                )

            simulator = await runtime.run(
                TuringSimulator,
                initial.controls.model_dump(),
                (config.preview_size, config.preview_size),
                initial.seed,
            )
            simulation_lock = asyncio.Lock()
            send_lock = asyncio.Lock()
            running = asyncio.Event()
            running.set()
            last_activity = monotonic()
            frame_id = 0
            controls_revision = 0

            await websocket.send_json(
                {
                    "type": "ready",
                    "protocol_version": 1,
                    "engine_version": ENGINE_VERSION,
                    "session_id": session_id,
                    "preview_size": config.preview_size,
                    "frame_rate": config.frame_rate,
                    "iteration": simulator.state.iteration,
                }
            )

            async def send_frame(frame: FrameResult, applied_revision: int) -> None:
                nonlocal frame_id, usage_frames, usage_bytes
                async with send_lock:
                    frame_id += 1
                    await websocket.send_json(
                        {
                            "type": "frame",
                            "frame_id": frame_id,
                            "iteration": frame.iteration,
                            "controls_revision": applied_revision,
                        }
                    )
                    await websocket.send_bytes(frame.content)
                    usage_frames += 1
                    usage_bytes += len(frame.content)
                    if usage_frames >= 100:
                        flush_session_usage()

            async def flush_usage_periodically() -> None:
                while True:
                    await asyncio.sleep(60)
                    flush_session_usage()

            async def receive_controls() -> None:
                nonlocal controls_revision, last_activity
                while True:
                    message = await _receive_message(websocket, config)
                    last_activity = monotonic()
                    if isinstance(message, StartMessage):
                        raise ClientProtocolError(
                            "already_started", "A session can only be started once."
                        )
                    if isinstance(message, PauseMessage):
                        running.clear()
                        continue
                    if isinstance(message, ResumeMessage):
                        running.set()
                        continue
                    if isinstance(message, StepMessage):
                        running.clear()
                        async with simulation_lock:
                            frame = await runtime.run(_encode_preview, simulator, 1)
                            frame_revision = controls_revision
                        runtime.metrics.observe_frame(
                            frame.step_seconds,
                            frame.encode_seconds,
                            len(frame.content),
                        )
                        await send_frame(frame, frame_revision)
                        continue

                    async with simulation_lock:
                        if isinstance(message, ControlsMessage):
                            await runtime.run(
                                simulator.update_controls,
                                message.controls.model_dump(),
                            )
                            if message.revision is not None:
                                controls_revision = message.revision
                        elif isinstance(message, ResetMessage):
                            await runtime.run(simulator.reset, message.seed)
                        elif isinstance(message, PerturbMessage):
                            await runtime.run(simulator.perturb, message.noise)

            async def send_frames() -> None:
                frame_interval = 1.0 / config.frame_rate
                while True:
                    if monotonic() - last_activity > config.idle_timeout_seconds:
                        await _send_error_and_close(
                            websocket,
                            "idle_timeout",
                            "Session timed out after "
                            f"{config.idle_timeout_seconds:g} seconds without input.",
                            1000,
                        )
                        return
                    if not running.is_set():
                        await asyncio.sleep(min(frame_interval, 0.1))
                        continue

                    frame_started = monotonic()
                    async with simulation_lock:
                        frame = await runtime.run(
                            _encode_preview, simulator, config.steps_per_frame
                        )
                        frame_revision = controls_revision
                    runtime.metrics.observe_frame(
                        frame.step_seconds, frame.encode_seconds, len(frame.content)
                    )
                    await send_frame(frame, frame_revision)
                    delay = frame_interval - (monotonic() - frame_started)
                    if delay > 0:
                        await asyncio.sleep(delay)

            tasks = {
                asyncio.create_task(receive_controls(), name=f"receive-{session_id}"),
                asyncio.create_task(send_frames(), name=f"frames-{session_id}"),
                asyncio.create_task(
                    flush_usage_periodically(), name=f"usage-{session_id}"
                ),
            }
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
        except (WebSocketDisconnect, asyncio.CancelledError):
            pass
        except ClientProtocolError as error:
            if accepted:
                await _send_error_and_close(
                    websocket, error.code, error.message, close_code=1008
                )
        except SimulationError:
            runtime.metrics.increment("numerical_failures")
            usage.increment("numerical_failures")
            logger.exception(
                "websocket simulation failed",
                extra={"event": "websocket_failed", "session_id": session_id},
            )
            if accepted:
                await _send_error_and_close(
                    websocket,
                    "simulation_failed",
                    "The simulation left its stable numerical range.",
                    close_code=1011,
                )
        except Exception:
            usage.increment("internal_errors")
            logger.exception(
                "websocket failed",
                extra={"event": "websocket_failed", "session_id": session_id},
            )
            if accepted:
                await _send_error_and_close(
                    websocket,
                    "internal_error",
                    "The simulation stopped unexpectedly.",
                    close_code=1011,
                )
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            flush_session_usage()
            usage.increment("live_sessions_finished")
            runtime.gate.release()
            runtime.metrics.increment("sessions_finished")
            logger.info(
                "websocket finished",
                extra={
                    "event": "websocket_finished",
                    "session_id": session_id,
                    "duration_seconds": monotonic() - started,
                },
            )

    return application


app = create_app()
