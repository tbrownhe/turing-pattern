from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from io import BytesIO
from time import monotonic
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from PIL import Image, PngImagePlugin
from starlette.websockets import WebSocketDisconnect

from app.api.schemas import (
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
from app.core.turing import SimulationError, TuringSimulator
from app.runtime import ComputeRuntime


logger = logging.getLogger(__name__)


class ClientProtocolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _encode_preview(simulator: TuringSimulator, steps: int) -> bytes:
    frame = simulator.step(steps=steps)
    output = BytesIO()
    Image.fromarray(frame).save(output, format="PNG", optimize=False)
    return output.getvalue()


def _render_png(payload: RenderRequest, config: Settings) -> bytes:
    simulator = TuringSimulator(
        payload.controls.model_dump(),
        shape=(config.render_size, config.render_size),
        seed=payload.seed,
    )
    frame = simulator.step(steps=config.render_steps)
    image = Image.fromarray(frame).convert("L")
    if config.render_upsample != 1:
        output_size = config.render_size * config.render_upsample
        image = image.resize((output_size, output_size), Image.Resampling.BICUBIC)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text(
        "TuringParams",
        payload.model_dump_json(indent=2),
    )
    output = BytesIO()
    image.save(output, format="PNG", pnginfo=metadata)
    return output.getvalue()


async def _receive_message(websocket: WebSocket, config: Settings):
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
    async def lifespan(application: FastAPI):
        runtime = ComputeRuntime(config)
        application.state.runtime = runtime
        try:
            yield
        finally:
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

    @application.post("/api/v1/generate", response_class=Response)
    async def generate(payload: RenderRequest, request: Request) -> Response:
        runtime: ComputeRuntime = request.app.state.runtime
        request_id = uuid4().hex
        if not await runtime.gate.acquire():
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "server_busy",
                    "message": "Render capacity is currently full. Please retry shortly.",
                },
                headers={"Retry-After": "2", "X-Request-ID": request_id},
            )

        logger.info("render_started request_id=%s", request_id)
        started = monotonic()
        try:
            content = await runtime.run(_render_png, payload, config)
        except SimulationError as error:
            raise HTTPException(
                status_code=422,
                detail={"code": "simulation_failed", "message": str(error)},
                headers={"X-Request-ID": request_id},
            ) from error
        finally:
            runtime.gate.release()

        logger.info(
            "render_completed request_id=%s duration_seconds=%.3f bytes=%d",
            request_id,
            monotonic() - started,
            len(content),
        )
        return Response(
            content=content,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "Content-Disposition": 'attachment; filename="turing-pattern.png"',
                "X-Request-ID": request_id,
            },
        )

    @application.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        runtime: ComputeRuntime = websocket.app.state.runtime
        session_id = uuid4().hex
        accepted = False
        tasks: set[asyncio.Task[None]] = set()

        if not _origin_is_allowed(websocket, config):
            await websocket.close(code=1008, reason="WebSocket origin is not allowed.")
            return

        if not await runtime.gate.acquire():
            await websocket.accept()
            await _send_error_and_close(
                websocket,
                "server_busy",
                "All simulation slots are busy. Please retry shortly.",
                1013,
            )
            return

        logger.info("websocket_started session_id=%s", session_id)
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
                    await websocket.send_bytes(frame)
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
        except Exception:
            logger.exception("websocket_failed session_id=%s", session_id)
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
            logger.info(
                "websocket_finished session_id=%s duration_seconds=%.3f",
                session_id,
                monotonic() - started,
            )

    return application


app = create_app()
