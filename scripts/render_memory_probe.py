"""Measure peak backend RSS while rendering the largest configured numerical grid."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Any

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


def request(
    url: str, *, payload: dict[str, Any] | None = None, method: str | None = None
) -> tuple[int, bytes, dict[str, str]]:
    body = json.dumps(payload).encode() if payload is not None else None
    http_request = urllib.request.Request(
        url,
        data=body,
        method=method or ("POST" if body is not None else "GET"),
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(http_request, timeout=10) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)
    except urllib.error.URLError as error:
        raise ConnectionError(f"could not reach {url}: {error.reason}") from error


def request_json(
    url: str, *, payload: dict[str, Any] | None = None, method: str | None = None
) -> tuple[int, dict[str, Any]]:
    status, body, _ = request(url, payload=payload, method=method)
    return status, json.loads(body)


def parse_metrics(raw: bytes) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in raw.decode().splitlines():
        if not line or line.startswith("#"):
            continue
        name, value = line.split(maxsplit=1)
        metrics[name] = float(value)
    return metrics


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3000")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--memory-limit-mb", type=float, default=2048.0)
    parser.add_argument("--max-memory-fraction", type=float, default=0.70)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=0.10)
    args = parser.parse_args()
    if args.memory_limit_mb <= 0 or not 0 < args.max_memory_fraction <= 1:
        parser.error("memory limit must be positive and fraction must be in (0, 1]")
    if not 10 <= args.timeout <= 900 or not 0.05 <= args.poll_interval <= 5:
        parser.error("timeout must be 10..900 seconds and polling 0.05..5 seconds")

    base_url = args.url.rstrip("/")
    payload = {
        "controls": CONTROLS,
        "seed": 5060,
        "width": 13.65,
        "height": 13.65,
        "unit": "in",
        "quality": "draft",
        "feature_scale": 1.0,
        "development_steps": 100,
        "framing": "crop",
    }
    plan_status, plan = request_json(f"{base_url}/api/v1/render-plans", payload=payload)
    if plan_status != 200 or not plan.get("accepted"):
        if plan_status == 404:
            raise SystemExit(
                "render-plan endpoint returned HTTP 404; rebuild the deployed backend "
                "from the commit containing P2.2 before running this probe"
            )
        raise SystemExit(f"maximum-grid plan was rejected: HTTP {plan_status} {plan}")

    submit_status, job = request_json(f"{base_url}/api/v1/renders", payload=payload)
    if submit_status != 202:
        raise SystemExit(f"render submission failed: HTTP {submit_status} {job}")

    rss_samples: list[float] = []
    lag_samples: list[float] = []
    health_samples: list[float] = []
    started = time.monotonic()
    try:
        while job["state"] in {"queued", "running"}:
            metrics_status, metrics_body, _ = request(args.metrics_url)
            if metrics_status != 200:
                raise RuntimeError(f"metrics returned HTTP {metrics_status}")
            metrics = parse_metrics(metrics_body)
            rss_samples.append(metrics["turing_process_resident_memory_bytes"])
            lag_samples.append(metrics["turing_event_loop_lag_seconds"])

            health_started = time.perf_counter()
            health_status, _, _ = request(f"{base_url}/healthz")
            health_samples.append(time.perf_counter() - health_started)
            if health_status != 200:
                raise RuntimeError(f"health returned HTTP {health_status}")

            status, job = request_json(f"{base_url}/api/v1/renders/{job['id']}")
            if status != 200:
                raise RuntimeError(f"job status returned HTTP {status}")
            if time.monotonic() - started > args.timeout:
                request(f"{base_url}/api/v1/renders/{job['id']}", method="DELETE")
                raise TimeoutError("maximum-grid probe exceeded its safety timeout")
            if job["state"] in {"queued", "running"}:
                time.sleep(args.poll_interval)
    except Exception:
        request(f"{base_url}/api/v1/renders/{job['id']}", method="DELETE")
        raise

    if job["state"] != "completed":
        raise SystemExit(f"maximum-grid render ended in {job['state']}: {job['error']}")
    artifact_status, artifact, headers = request(f"{base_url}{job['artifact_url']}")
    artifact_content_type = next(
        (value for name, value in headers.items() if name.lower() == "content-type"),
        "",
    )
    if (
        artifact_status != 200
        or not artifact_content_type.startswith("image/png")
        or not artifact.startswith(b"\x89PNG\r\n\x1a\n")
    ):
        raise SystemExit("maximum-grid artifact was not a valid PNG response")

    memory_limit_bytes = args.memory_limit_mb * 1024 * 1024
    peak_rss = max(rss_samples, default=0.0)
    result = {
        "label": "maximum-grid-render-memory",
        "job_id": job["id"],
        "simulation_grid": [plan["simulation_width"], plan["simulation_height"]],
        "simulation_pixels": plan["simulation_pixels"],
        "output_pixels": [plan["output_width"], plan["output_height"]],
        "steps": job["requested_steps"],
        "duration_seconds": round(time.monotonic() - started, 3),
        "rss_samples": len(rss_samples),
        "peak_rss_bytes": round(peak_rss),
        "container_memory_limit_bytes": round(memory_limit_bytes),
        "peak_memory_fraction": round(peak_rss / memory_limit_bytes, 4),
        "max_event_loop_lag_ms": round(max(lag_samples, default=0.0) * 1000, 2),
        "health_mean_ms": round(
            statistics.fmean(health_samples) * 1000 if health_samples else 0.0, 2
        ),
        "health_p95_ms": round(percentile(health_samples, 0.95) * 1000, 2),
        "artifact_bytes": len(artifact),
        "below_memory_budget": peak_rss
        <= memory_limit_bytes * args.max_memory_fraction,
    }
    print(json.dumps(result, indent=2))
    if not result["below_memory_budget"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
