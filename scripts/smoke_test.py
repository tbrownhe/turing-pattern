"""Smoke-test the local same-origin Compose stack.

Run after ``docker compose -f docker-compose.local.yml up -d`` with:

    uv run --with websockets python scripts/smoke_test.py
"""

from __future__ import annotations

import asyncio
import json
from urllib.request import Request, urlopen

import websockets


BASE_URL = "http://127.0.0.1:3000"
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


def check_http() -> None:
    with urlopen(f"{BASE_URL}/", timeout=5) as response:
        body = response.read()
        assert response.status == 200
        assert b"Gray-Scott Pattern Lab" in body
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]

    with urlopen(f"{BASE_URL}/healthz", timeout=5) as response:
        assert json.load(response) == {"status": "ok"}

    payload = json.dumps({"seed": 11, "controls": CONTROLS}).encode()
    request = Request(
        f"{BASE_URL}/api/v1/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        assert response.status == 200
        assert response.headers["Content-Type"] == "image/png"
        assert response.read(8) == b"\x89PNG\r\n\x1a\n"


async def check_websocket() -> None:
    async with websockets.connect(
        "ws://127.0.0.1:3000/ws",
        origin=BASE_URL,
        open_timeout=5,
    ) as websocket:
        await websocket.send(
            json.dumps(
                {
                    "type": "start",
                    "protocol_version": 1,
                    "seed": 11,
                    "controls": CONTROLS,
                }
            )
        )
        ready = json.loads(await asyncio.wait_for(websocket.recv(), timeout=10))
        assert ready["type"] == "ready"
        frame = await asyncio.wait_for(websocket.recv(), timeout=10)
        assert isinstance(frame, bytes)
        assert frame.startswith(b"\x89PNG\r\n\x1a\n")
        await websocket.send(json.dumps({"type": "pause"}))


def main() -> None:
    check_http()
    asyncio.run(check_websocket())
    print("Local HTTP, render, security-header, and WebSocket smoke tests passed.")


if __name__ == "__main__":
    main()
