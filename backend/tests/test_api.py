import json
from dataclasses import replace
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from app.api.main import create_app
from app.config import settings

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
ORIGIN = "http://testserver"


@pytest.fixture
def client(tmp_path):
    config = replace(
        settings,
        allowed_origins=(ORIGIN,),
        preview_size=8,
        render_size=8,
        render_steps=2,
        render_upsample=2,
        steps_per_frame=1,
        frame_rate=5,
        max_compute_jobs=1,
        compute_workers=1,
        max_compute_waiters=0,
        render_data_dir=str(tmp_path / "renders"),
        max_render_job_history=8,
        render_chunk_steps=10,
    )
    with TestClient(create_app(config)) as test_client:
        yield test_client


def test_health_does_not_consume_compute_capacity(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    readiness = client.get("/readyz").json()
    assert readiness["active_compute_jobs"] == 0
    assert readiness["compute_capacity"] == 1


def test_websocket_rejects_an_unapproved_browser_origin(client):
    with pytest.raises(WebSocketDisconnect) as caught:
        with client.websocket_connect(
            "/ws", headers={"origin": "https://evil.invalid"}
        ):
            pass

    assert caught.value.code == 1008
    assert client.get("/readyz").json()["active_compute_jobs"] == 0


def test_websocket_rejects_constructor_options(client):
    with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as websocket:
        websocket.send_json(
            {
                "type": "start",
                "protocol_version": 1,
                "controls": CONTROLS,
                "shape": [50_000, 50_000],
            }
        )
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["error"]["code"] == "invalid_message"
    assert client.get("/readyz").json()["active_compute_jobs"] == 0


def test_websocket_starts_and_produces_a_png_frame(client):
    with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as websocket:
        websocket.send_json(
            {
                "type": "start",
                "protocol_version": 1,
                "controls": CONTROLS,
                "seed": 7,
            }
        )
        ready = websocket.receive_json()
        metadata = websocket.receive_json()
        frame = websocket.receive_bytes()
        websocket.send_json({"type": "pause"})

    assert ready["type"] == "ready"
    assert ready["engine_version"] == "2.0.0"
    assert ready["preview_size"] == 8
    assert metadata == {"type": "frame", "frame_id": 1, "iteration": 1}
    assert frame.startswith(b"\x89PNG\r\n\x1a\n")


def test_websocket_can_advance_one_iteration_while_paused(client):
    with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as websocket:
        websocket.send_json(
            {
                "type": "start",
                "protocol_version": 1,
                "controls": CONTROLS,
                "seed": 7,
            }
        )
        assert websocket.receive_json()["type"] == "ready"
        assert websocket.receive_json()["iteration"] == 1
        websocket.receive_bytes()
        websocket.send_json({"type": "pause"})
        websocket.send_json({"type": "step"})
        stepped_metadata = websocket.receive_json()
        stepped_frame = websocket.receive_bytes()

    assert stepped_metadata["iteration"] == 2
    assert stepped_frame.startswith(b"\x89PNG\r\n\x1a\n")


def test_websocket_rejects_excess_sessions_without_queueing(client):
    start = {
        "type": "start",
        "protocol_version": 1,
        "controls": CONTROLS,
        "seed": 7,
    }
    with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as first:
        first.send_json(start)
        assert first.receive_json()["type"] == "ready"

        with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as second:
            response = second.receive_json()

    assert response["error"]["code"] == "server_busy"
    assert client.get("/readyz").json()["active_compute_jobs"] == 0


def test_generate_is_post_only_and_validated(client):
    assert client.get("/api/v1/generate").status_code == 405
    invalid = client.post(
        "/api/v1/generate",
        json={"controls": {**CONTROLS, "F1": "not-a-number"}},
    )
    assert invalid.status_code == 422

    response = client.post("/api/v1/generate", json={"controls": CONTROLS, "seed": 9})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-turing-engine"] == "2.0.0"
    assert response.content.startswith(b"\x89PNG\r\n\x1a\n")


def test_render_plan_resolves_physical_dimensions_and_rejects_oversized_work(client):
    payload = {
        "controls": CONTROLS,
        "seed": 9,
        "width": 6.0,
        "height": 6.0,
        "unit": "in",
        "quality": "studio",
        "feature_scale": 1.0,
        "development_steps": 5000,
        "framing": "crop",
    }

    accepted = client.post("/api/v1/render-plans", json=payload).json()
    oversized = client.post(
        "/api/v1/render-plans", json={**payload, "width": 12.0, "height": 12.0}
    ).json()

    assert accepted["accepted"] is True
    assert accepted["output_width"] == 1800
    assert accepted["simulation_width"] == 900
    assert accepted["bicubic_upsample"] == 2
    assert accepted["scale_model_status"] == "reference-validated"
    assert accepted["estimated_seconds_high"] > accepted["estimated_seconds_low"]
    assert oversized["accepted"] is False
    assert oversized["issues"]


def test_render_job_completes_persists_metadata_and_serves_an_artifact(client):
    payload = {
        "controls": CONTROLS,
        "seed": 19,
        "width": 0.1,
        "height": 0.1,
        "unit": "in",
        "quality": "draft",
        "feature_scale": 1.0,
        "development_steps": 100,
        "framing": "crop",
    }

    queued = client.post("/api/v1/renders", json=payload)
    assert queued.status_code == 202
    assert queued.headers["location"].startswith("/api/v1/renders/")
    job = queued.json()
    for _ in range(100):
        job = client.get(f"/api/v1/renders/{job['id']}").json()
        if job["state"] in {"completed", "failed"}:
            break

    assert job["state"] == "completed", job
    assert job["progress_steps"] == 100
    assert job["artifact_available"] is True
    artifact = client.get(job["artifact_url"])
    assert artifact.status_code == 200
    assert artifact.headers["content-type"] == "image/png"
    image = Image.open(BytesIO(artifact.content))
    metadata = json.loads(image.text["TuringParams"])
    assert metadata["actual_steps"] == 100
    assert metadata["recipe"]["seed"] == 19
    assert metadata["plan"]["output_width"] == 15

    repeated = client.post("/api/v1/renders", json=payload).json()
    for _ in range(100):
        repeated = client.get(f"/api/v1/renders/{repeated['id']}").json()
        if repeated["state"] in {"completed", "failed"}:
            break
    repeated_artifact = client.get(repeated["artifact_url"])
    assert repeated["state"] == "completed"
    assert repeated_artifact.content == artifact.content


def test_render_job_can_be_cancelled_while_waiting_for_compute(client):
    payload = {
        "controls": CONTROLS,
        "seed": 19,
        "width": 0.1,
        "height": 0.1,
        "unit": "in",
        "quality": "draft",
        "feature_scale": 1.0,
        "development_steps": 100,
        "framing": "crop",
    }
    with client.websocket_connect("/ws", headers={"origin": ORIGIN}) as websocket:
        websocket.send_json(
            {
                "type": "start",
                "protocol_version": 1,
                "controls": CONTROLS,
                "seed": 7,
            }
        )
        assert websocket.receive_json()["type"] == "ready"
        queued = client.post("/api/v1/renders", json=payload).json()
        cancelled = client.delete(f"/api/v1/renders/{queued['id']}")

    assert cancelled.status_code == 202
    assert cancelled.json()["state"] == "cancelled"
    assert client.get(f"/api/v1/renders/{queued['id']}/artifact").status_code == 409


def test_uncalibrated_feature_scale_cannot_enter_the_queue(client):
    response = client.post(
        "/api/v1/renders",
        json={
            "controls": CONTROLS,
            "seed": 19,
            "width": 0.1,
            "height": 0.1,
            "unit": "in",
            "quality": "draft",
            "feature_scale": 2.0,
            "development_steps": 100,
            "framing": "crop",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "render_plan_rejected"


def test_terminal_render_history_is_bounded(client):
    payload = {
        "controls": CONTROLS,
        "seed": 23,
        "width": 0.1,
        "height": 0.1,
        "unit": "in",
        "quality": "draft",
        "feature_scale": 1.0,
        "development_steps": 100,
        "framing": "crop",
    }
    completed_ids = []
    for seed in range(9):
        job = client.post("/api/v1/renders", json={**payload, "seed": seed}).json()
        for _ in range(100):
            job = client.get(f"/api/v1/renders/{job['id']}").json()
            if job["state"] in {"completed", "failed"}:
                break
        assert job["state"] == "completed", job
        completed_ids.append(job["id"])

    assert client.get(f"/api/v1/renders/{completed_ids[0]}").status_code == 404
    assert client.get(f"/api/v1/renders/{completed_ids[-1]}").status_code == 200


def test_time_study_runs_one_bounded_simulation_and_returns_ordered_images(client):
    response = client.post(
        "/api/v1/time-studies",
        json={
            "controls": CONTROLS,
            "seed": 3,
            "checkpoints": [100, 200],
        },
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    body = response.json()
    assert body["simulation_size"] == 8
    assert [checkpoint["steps"] for checkpoint in body["checkpoints"]] == [100, 200]
    assert all(
        checkpoint["image_url"].startswith("data:image/png;base64,")
        for checkpoint in body["checkpoints"]
    )
    metrics = client.get("/metrics").text
    assert "turing_time_studies_started_total 1" in metrics
    assert "turing_time_studies_finished_total 1" in metrics


def test_generate_is_reproducible_and_embeds_a_complete_recipe(client):
    payload = {"controls": CONTROLS, "seed": 19}

    first = client.post("/api/v1/generate", json=payload)
    second = client.post("/api/v1/generate", json=payload)
    image = Image.open(BytesIO(first.content))
    recipe = json.loads(image.text["TuringParams"])

    assert first.content == second.content
    assert recipe == {
        "engine_version": "2.0.0",
        "boundary": "periodic",
        "dtype": "float32",
        "simulation_size": 8,
        "steps": 2,
        "upsample": 2,
        "controls": CONTROLS,
        "seed": 19,
    }


def test_metrics_report_work_without_high_cardinality_ids(client):
    client.post("/api/v1/generate", json={"controls": CONTROLS, "seed": 2})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "turing_renders_started_total 1" in response.text
    assert "turing_renders_finished_total 1" in response.text
    assert "turing_render_simulation_seconds_total " in response.text
    assert "turing_process_resident_memory_bytes " in response.text
    assert "request_id" not in response.text
