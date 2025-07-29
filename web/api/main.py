import base64
from io import BytesIO

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

from src.turing import turing_pattern

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
