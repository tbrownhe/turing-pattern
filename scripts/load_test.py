"""Bounded live/render load probe for a running Turing Pattern stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlparse

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

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


@dataclass(slots=True)
class Results:
    admitted: int = 0
    rejected: int = 0
    frames: int = 0
    disconnects: int = 0
    errors: list[str] = field(default_factory=list)
    health_seconds: list[float] = field(default_factory=list)
    render_statuses: list[int] = field(default_factory=list)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def fetch(url: str, *, payload: bytes | None = None) -> int:
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"} if payload else {},
        method="POST" if payload else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            response.read()
            return response.status
    except urllib.error.HTTPError as error:
        error.read()
        return error.code


async def health_probe(base_url: str, duration: float, results: Results) -> None:
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        started = time.perf_counter()
        try:
            status = await asyncio.to_thread(fetch, f"{base_url}/healthz")
            if status != 200:
                results.errors.append(f"health returned HTTP {status}")
        except Exception as error:  # noqa: BLE001 - this is a diagnostic tool
            results.errors.append(f"health failed: {error}")
        results.health_seconds.append(time.perf_counter() - started)
        await asyncio.sleep(0.25)


async def exercise_session(
    ws_url: str,
    duration: float,
    seed: int,
    message_rate: float,
    results: Results,
) -> None:
    try:
        async with connect(ws_url, max_size=1_048_576) as websocket:
            await websocket.send(
                json.dumps(
                    {
                        "type": "start",
                        "protocol_version": 1,
                        "controls": CONTROLS,
                        "seed": seed,
                    }
                )
            )
            initial = await asyncio.wait_for(websocket.recv(), timeout=5)
            if isinstance(initial, bytes):
                results.errors.append("received a frame before the ready message")
                return
            message = json.loads(initial)
            if message.get("type") == "error":
                results.rejected += 1
                return
            if message.get("type") != "ready":
                results.errors.append(f"unexpected initial message: {message!r}")
                return

            results.admitted += 1

            async def send_controls() -> None:
                if message_rate <= 0:
                    return
                interval = 1.0 / message_rate
                while True:
                    await websocket.send(
                        json.dumps({"type": "controls", "controls": CONTROLS})
                    )
                    await asyncio.sleep(interval)

            sender = asyncio.create_task(send_controls())
            deadline = time.monotonic() + duration
            try:
                while time.monotonic() < deadline:
                    received = await asyncio.wait_for(websocket.recv(), timeout=2)
                    if isinstance(received, bytes):
                        results.frames += 1
            finally:
                sender.cancel()
                await asyncio.gather(sender, return_exceptions=True)
    except ConnectionClosed as error:
        if error.code == 1013:
            results.rejected += 1
        else:
            results.disconnects += 1
    except Exception as error:  # noqa: BLE001 - preserve all probe failures
        results.errors.append(f"session {seed}: {error}")


async def disconnect_storm(ws_url: str, count: int, results: Results) -> None:
    for seed in range(count):
        try:
            async with connect(ws_url) as websocket:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "start",
                            "protocol_version": 1,
                            "controls": CONTROLS,
                            "seed": seed,
                        }
                    )
                )
        except Exception as error:  # noqa: BLE001 - preserve probe failures
            results.errors.append(f"storm connection {seed}: {error}")


async def render_probe(
    base_url: str, count: int, delay: float, results: Results
) -> None:
    await asyncio.sleep(delay)
    payload = json.dumps({"controls": CONTROLS, "seed": 7}).encode()
    calls = [
        asyncio.to_thread(fetch, f"{base_url}/api/v1/generate", payload=payload)
        for _ in range(count)
    ]
    if calls:
        responses = await asyncio.gather(*calls, return_exceptions=True)
        for response in responses:
            if isinstance(response, BaseException):
                results.errors.append(f"render failed: {response}")
            else:
                results.render_statuses.append(response)


async def run(args: argparse.Namespace) -> Results:
    base_url = args.url.rstrip("/")
    parsed = urlparse(base_url)
    ws_scheme = "wss" if parsed.scheme == "https" else "ws"
    ws_url = f"{ws_scheme}://{parsed.netloc}/ws"
    results = Results()

    tasks = [
        asyncio.create_task(health_probe(base_url, args.duration + 2, results)),
        asyncio.create_task(
            render_probe(base_url, args.renders, args.render_delay, results)
        ),
        *[
            asyncio.create_task(
                exercise_session(
                    ws_url,
                    args.duration,
                    seed,
                    args.message_rate,
                    results,
                )
            )
            for seed in range(args.clients + args.excess)
        ],
    ]
    await asyncio.gather(*tasks)
    await disconnect_storm(ws_url, args.disconnect_storm, results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--clients", type=int, default=2)
    parser.add_argument("--excess", type=int, default=1)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--renders", type=int, default=2)
    parser.add_argument(
        "--render-delay",
        type=float,
        default=1.0,
        help="Seconds to let live sessions acquire capacity before render probes.",
    )
    parser.add_argument("--disconnect-storm", type=int, default=5)
    parser.add_argument(
        "--message-rate",
        type=float,
        default=0.0,
        help="Valid control messages per second per admitted client (max 100).",
    )
    args = parser.parse_args()
    if args.clients < 0 or args.excess < 0 or not 1 <= args.duration <= 300:
        parser.error("client counts must be non-negative and duration must be 1..300")
    if not 0 <= args.renders <= 20 or not 0 <= args.disconnect_storm <= 100:
        parser.error("renders must be 0..20 and disconnect-storm must be 0..100")
    if not 0 <= args.render_delay <= args.duration:
        parser.error("render-delay must be between zero and duration")
    if not 0 <= args.message_rate <= 100:
        parser.error("message-rate must be 0..100")

    results = asyncio.run(run(args))
    report = {
        "admitted_sessions": results.admitted,
        "rejected_sessions": results.rejected,
        "frames": results.frames,
        "disconnects": results.disconnects,
        "render_statuses": results.render_statuses,
        "health_samples": len(results.health_seconds),
        "health_mean_ms": round(statistics.fmean(results.health_seconds) * 1000, 2),
        "health_p95_ms": round(percentile(results.health_seconds, 0.95) * 1000, 2),
        "health_max_ms": round(max(results.health_seconds, default=0.0) * 1000, 2),
        "errors": results.errors,
    }
    print(json.dumps(report, indent=2))
    if results.errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
