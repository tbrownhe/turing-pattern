from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from io import BytesIO
from time import monotonic, perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from PIL import Image, PngImagePlugin
from pydantic import ValidationError
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import (
    ClientMessage,
    ControlsMessage,
    PauseMessage,
    PerturbMessage,
    RenderRequest,
    ResetMessage,
    ResumeMessage,
    StartMessage,
    parse_client_message,
)
from app.config import Settings, settings
from app.core.engine import SimulationError, TuringSimulator
from app.core.models import ENGINE_VERSION
from app.observability import configure_logging
from app.runtime import ComputeRuntime

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


@dataclass(frozen=True, slots=True)
class RenderResult:
    content: bytes
    simulation_seconds: float
    encode_seconds: float


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
    )


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
        runtime = ComputeRuntime(config)
        application.state.runtime = runtime
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
            runtime.close()

    application = FastAPI(
        title="Turing Pattern API",
        version="1.0.0",
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
        allow_methods=["GET", "POST", "OPTIONS"],
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

    @application.post("/api/v1/generate", response_class=Response)
    async def generate(payload: RenderRequest, request: Request) -> Response:
        runtime: ComputeRuntime = request.app.state.runtime
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
        logger.info(
            "render started",
            extra={"event": "render_started", "request_id": request_id},
        )
        started = monotonic()
        try:
            result = await runtime.run(_render_png, payload, config)
        except SimulationError as error:
            runtime.metrics.increment("numerical_failures")
            raise HTTPException(
                status_code=422,
                detail={"code": "simulation_failed", "message": str(error)},
                headers={"X-Request-ID": request_id},
            ) from error
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
        session_id = uuid4().hex
        accepted = False
        tasks: set[asyncio.Task[None]] = set()

        if not _origin_is_allowed(websocket, config):
            runtime.metrics.increment("sessions_rejected")
            await websocket.close(code=1008, reason="WebSocket origin is not allowed.")
            return

        if not await runtime.gate.acquire():
            runtime.metrics.increment("sessions_rejected")
            await websocket.accept()
            await _send_error_and_close(
                websocket,
                "server_busy",
                "All simulation slots are busy. Please retry shortly.",
                1013,
            )
            return

        runtime.metrics.increment("sessions_started")
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
            running = asyncio.Event()
            running.set()
            last_activity = monotonic()

            await websocket.send_json(
                {
                    "type": "ready",
                    "protocol_version": 1,
                    "engine_version": ENGINE_VERSION,
                    "session_id": session_id,
                    "preview_size": config.preview_size,
                    "frame_rate": config.frame_rate,
                }
            )

            async def receive_controls() -> None:
                nonlocal last_activity
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

                    async with simulation_lock:
                        if isinstance(message, ControlsMessage):
                            await runtime.run(
                                simulator.update_controls,
                                message.controls.model_dump(),
                            )
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
                    runtime.metrics.observe_frame(
                        frame.step_seconds, frame.encode_seconds, len(frame.content)
                    )
                    await websocket.send_bytes(frame.content)
                    delay = frame_interval - (monotonic() - frame_started)
                    if delay > 0:
                        await asyncio.sleep(delay)

            tasks = {
                asyncio.create_task(receive_controls(), name=f"receive-{session_id}"),
                asyncio.create_task(send_frames(), name=f"frames-{session_id}"),
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
