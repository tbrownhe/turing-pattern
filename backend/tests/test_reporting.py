import json
from datetime import date
from io import BytesIO

from app import reporting
from app.reporting import ReportSettings, build_digest, cloudflare_analytics
from app.usage import COUNTERS


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self):
        return BytesIO(self.payload).read()


def report_settings(**overrides):
    values = {
        "data_dir": "/tmp/turing",
        "timezone_name": "America/Los_Angeles",
        "hostname": "turing.example.com",
        "cloudflare_zone_id": "zone-id",
        "cloudflare_token": "token",
        "report_to": "owner@example.com",
        "report_from": "reports@example.com",
        "smtp_host": "smtp.example.com",
        "smtp_port": 587,
        "smtp_security": "starttls",
        "smtp_user": "owner",
        "smtp_password": "password",
    }
    values.update(overrides)
    return ReportSettings(**values)


def test_cloudflare_analytics_sums_aggregate_groups():
    captured = {}

    def opener(request, timeout):
        captured["body"] = json.loads(request.data)
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "data": {
                    "viewer": {
                        "zones": [
                            {
                                "httpRequestsAdaptiveGroups": [
                                    {
                                        "count": 20,
                                        "sum": {
                                            "visits": 7,
                                            "edgeResponseBytes": 2048,
                                        },
                                    },
                                    {
                                        "count": 5,
                                        "sum": {
                                            "visits": 2,
                                            "edgeResponseBytes": 512,
                                        },
                                    },
                                ]
                            }
                        ]
                    }
                }
            }
        )

    result = cloudflare_analytics(report_settings(), date(2026, 7, 19), opener=opener)

    assert result == {
        "status": "available",
        "visits": 9,
        "requests": 25,
        "bytes": 2560,
    }
    variables = captured["body"]["variables"]
    assert variables["start"] == "2026-07-19T07:00:00Z"
    assert variables["end"] == "2026-07-20T07:00:00Z"
    assert captured["timeout"] == 20


def test_missing_cloudflare_config_does_not_block_application_digest():
    result = cloudflare_analytics(
        report_settings(cloudflare_token=None), date(2026, 7, 19)
    )
    assert result["status"] == "unavailable"


def test_digest_contains_only_daily_aggregate_fields():
    app = {"day": "2026-07-19", **dict.fromkeys(COUNTERS, 0)}
    app.update(
        {
            "live_sessions_admitted": 4,
            "live_session_seconds": 300,
            "live_frame_bytes": 4096,
            "render_requests": 2,
            "render_completed": 1,
        }
    )
    body = build_digest(
        date(2026, 7, 19),
        app,
        {"status": "available", "visits": 3, "requests": 8, "bytes": 1024},
        "turing.example.com",
    )

    assert "Estimated visits: 3" in body
    assert "Sessions admitted: 4" in body
    assert "Live time: 5.0 minutes" in body
    assert "Requests: 2" in body
    assert "no IPs, user agents, recipes" in body


def test_email_mode_does_not_open_or_claim_the_usage_store(monkeypatch, tmp_path):
    captured = {}

    def fake_send(config, day, body, *, subject=None):
        captured.update(
            {"to": config.report_to, "day": day, "body": body, "subject": subject}
        )

    monkeypatch.setenv("TURING_RENDER_DATA_DIR", str(tmp_path / "unused"))
    monkeypatch.setenv("TURING_REPORT_TO", "owner@example.com")
    monkeypatch.setenv("TURING_REPORT_FROM", "reports@example.com")
    monkeypatch.setenv("TURING_SMTP_HOST", "smtp.example.com")
    monkeypatch.delenv("TURING_SMTP_USER", raising=False)
    monkeypatch.setattr(reporting, "send_email", fake_send)

    assert reporting.main(["--test-email", "--date", "2026-07-19"]) == 0
    assert captured == {
        "to": "owner@example.com",
        "day": date(2026, 7, 19),
        "body": "Turing Pattern reporting SMTP configuration succeeded.\n",
        "subject": "Turing Pattern reporting test",
    }
    assert not (tmp_path / "unused").exists()
