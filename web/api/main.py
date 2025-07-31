import asyncio
import time
from io import BytesIO

from fastapi import FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from src.turing import TuringSimulator, turing_pattern

MAX_SIMS = 4
MAX_IDLE = 600
sim_semaphore = asyncio.Semaphore(MAX_SIMS)

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/generate")
def generate(
    F1: float = Query(...),
    F2: float = Query(...),
    K1: float = Query(...),
    K2: float = Query(...),
    Du1: float = Query(...),
    Du2: float = Query(...),
    Dv1: float = Query(...),
    Dv2: float = Query(...),
):
    pattern = turing_pattern(
        w=256,
        h=256,
        F_ctrl=[0, 1],
        F_vals=[F1, F2],
        F_axis="x",
        k_ctrl=[0, 1],
        k_vals=[K1, K2],
        k_axis="x",
        Du_ctrl=[0, 1],
        Du_vals=[Du1, Du2],
        Du_axis="y",
        Dv_ctrl=[0, 1],
        Dv_vals=[Dv1, Dv2],
        Dv_axis="y",
        steps=5000,
        upsample=2,
    )
    img = Image.fromarray(pattern).convert("L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    try:
        await asyncio.wait_for(sim_semaphore.acquire(), timeout=0.5)
    except asyncio.TimeoutError:
        await websocket.accept()
        await websocket.send_json(
            {
                "error": f"Server busy. Maximum of {MAX_SIMS} simulations has been reached."
            }
        )
        await websocket.close()
        return

    await websocket.accept()
    last_time = time.time()

    try:
        # Get initial slider positions and kick off sim class
        msg = await websocket.receive_json()
        sim = TuringSimulator(**msg)

        # Create async task for updating control maps
        async def receive_controls():
            nonlocal last_time
            while True:
                msg = await websocket.receive_json()
                if msg.get("type") == "seed":
                    sim.seed()
                else:
                    sim.update_controls(msg)

                # Reset timer if user provides input
                last_time = time.time()

        receiver = asyncio.create_task(receive_controls())

        # Serve simulation to websocket
        try:
            while True:
                # Kill sessions that have been running without input for >5min
                if time.time() - last_time > MAX_IDLE:
                    await websocket.send_json(
                        {"error": f"Session timed out after {MAX_IDLE} seconds."}
                    )
                    await websocket.close()
                    break

                frame = sim.step(steps=25)
                buf = BytesIO()
                Image.fromarray(frame).save(buf, format="PNG")
                buf.seek(0)
                await websocket.send_bytes(buf.read())
                await asyncio.sleep(0.05)
        except WebSocketDisconnect:
            pass
        finally:
            receiver.cancel()
    finally:
        sim_semaphore.release()
