from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from io import BytesIO
from PIL import Image
import base64
from src.turing import turing_pattern

app = FastAPI()


@app.get("/generate")
def generate(F: float = Query(...), k: float = Query(...)):
    pattern = turing_pattern(
        F_ctrl=[0, 1], F_vals=[F, F], k_ctrl=[0, 1], k_vals=[k, k], steps=5000
    )
    img = Image.fromarray(pattern).convert("L")
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
