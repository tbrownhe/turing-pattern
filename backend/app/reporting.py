from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from email.message import EmailMessage
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from app.usage import UsageStore

CLOUDFLARE_GRAPHQL_URL = "https://api.cloudflare.com/client/v4/graphql"


def _read_secret(path: str | None) -> str | None:
    if not path:
        return None
    secret_path = Path(path)
    if not secret_path.is_file():
        return None
    return secret_path.read_text(encoding="utf-8").strip()


@dataclass(frozen=True, slots=True)
class ReportSettings:
    data_dir: str
    timezone_name: str
    hostname: str | None
    cloudflare_zone_id: str | None
    cloudflare_token: str | None
    report_to: str | None
    report_from: str | None
    smtp_host: str | None
    smtp_port: int
    smtp_security: str
    smtp_user: str | None
    smtp_password: str | None

    @classmethod
    def from_env(cls) -> ReportSettings:
        timezone_name = os.getenv("TURING_REPORT_TIMEZONE", "America/Los_Angeles")
        ZoneInfo(timezone_name)
        security = os.getenv("TURING_SMTP_SECURITY", "starttls").lower()
        if security not in {"starttls", "ssl", "plain"}:
            raise ValueError("TURING_SMTP_SECURITY must be starttls, ssl, or plain")
        return cls(
            data_dir=os.getenv("TURING_RENDER_DATA_DIR", "/var/lib/turing"),
            timezone_name=timezone_name,
            hostname=os.getenv("TURING_REPORT_HOSTNAME") or os.getenv("DOMAIN"),
            cloudflare_zone_id=os.getenv("TURING_CLOUDFLARE_ZONE_ID"),
            cloudflare_token=_read_secret(
                os.getenv(
                    "TURING_CLOUDFLARE_API_TOKEN_FILE",
                    "/run/secrets/turing-reporting/cloudflare-token",
                )
            ),
            report_to=os.getenv("TURING_REPORT_TO"),
            report_from=os.getenv("TURING_REPORT_FROM"),
            smtp_host=os.getenv("TURING_SMTP_HOST"),
            smtp_port=int(os.getenv("TURING_SMTP_PORT", "587")),
            smtp_security=security,
            smtp_user=os.getenv("TURING_SMTP_USER"),
            smtp_password=_read_secret(
                os.getenv(
                    "TURING_SMTP_PASSWORD_FILE",
                    "/run/secrets/turing-reporting/smtp-password",
                )
            ),
        )


def _utc_range(day: date, timezone_name: str) -> tuple[str, str]:
    zone = ZoneInfo(timezone_name)
    start = datetime.combine(day, time.min, zone).astimezone(UTC)
    end = datetime.combine(day + timedelta(days=1), time.min, zone).astimezone(UTC)
    return start.isoformat().replace("+00:00", "Z"), end.isoformat().replace(
        "+00:00", "Z"
    )


def cloudflare_analytics(
    config: ReportSettings,
    day: date,
    *,
    opener: Any = urllib.request.urlopen,
) -> dict[str, int | str]:
    if not (config.cloudflare_zone_id and config.cloudflare_token and config.hostname):
        return {"status": "unavailable", "detail": "Cloudflare is not configured."}

    start, end = _utc_range(day, config.timezone_name)
    query = """
    query DailyUsage(
      $zoneTag: string!, $hostname: string!, $start: Time!, $end: Time!
    ) {
      viewer {
        zones(filter: {zoneTag: $zoneTag}) {
          httpRequestsAdaptiveGroups(
            limit: 1000
            filter: {
              datetime_geq: $start
              datetime_lt: $end
              clientRequestHTTPHost: $hostname
              requestSource: "eyeball"
            }
          ) {
            count
            sum { visits edgeResponseBytes }
          }
        }
      }
    }
    """
    body = json.dumps(
        {
            "query": query,
            "variables": {
                "zoneTag": config.cloudflare_zone_id,
                "hostname": config.hostname,
                "start": start,
                "end": end,
            },
        }
    ).encode()
    request = urllib.request.Request(
        CLOUDFLARE_GRAPHQL_URL,
        data=body,
        headers={
            "Authorization": f"Bearer {config.cloudflare_token}",
            "Content-Type": "application/json",
            "User-Agent": "turing-pattern-daily-report/1",
        },
        method="POST",
    )
    with opener(request, timeout=20) as response:
        result = json.loads(response.read())
    if result.get("errors"):
        messages = "; ".join(
            str(error.get("message", error)) for error in result["errors"]
        )
        raise RuntimeError(f"Cloudflare GraphQL error: {messages}")
    zones = result.get("data", {}).get("viewer", {}).get("zones", [])
    if not zones:
        raise RuntimeError("Cloudflare returned no matching zone")
    groups = zones[0].get("httpRequestsAdaptiveGroups", [])
    return {
        "status": "available",
        "visits": sum(int(group.get("sum", {}).get("visits") or 0) for group in groups),
        "requests": sum(int(group.get("count") or 0) for group in groups),
        "bytes": sum(
            int(group.get("sum", {}).get("edgeResponseBytes") or 0) for group in groups
        ),
    }


def _format_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def build_digest(
    day: date,
    app: dict[str, int | float | str],
    cloudflare: dict[str, int | str],
    hostname: str | None,
) -> str:
    if cloudflare["status"] == "available":
        edge_lines = [
            f"Estimated visits: {cloudflare['visits']:,}",
            f"HTTP requests: {cloudflare['requests']:,}",
            f"Edge transfer: {_format_bytes(float(cloudflare['bytes']))}",
        ]
    else:
        edge_lines = [
            "Cloudflare analytics: unavailable",
            f"Reason: {cloudflare.get('detail', 'unknown error')}",
        ]
    app_errors = int(app["numerical_failures"]) + int(app["internal_errors"])
    lines = [
        f"Turing Pattern daily usage — {day.isoformat()}",
        f"Host: {hostname or 'not configured'}",
        "",
        "Public traffic (Cloudflare aggregate)",
        *edge_lines,
        "",
        "Live simulator",
        f"Sessions admitted: {int(app['live_sessions_admitted']):,}",
        f"Sessions finished: {int(app['live_sessions_finished']):,}",
        f"Sessions rejected: {int(app['live_sessions_rejected']):,}",
        f"Live time: {float(app['live_session_seconds']) / 60:.1f} minutes",
        f"Frames sent: {int(app['live_frames']):,}",
        f"Approx. stream bytes: {_format_bytes(float(app['live_frame_bytes']))}",
        f"Peak compute slots used: {int(app['peak_compute_active']):,}",
        "",
        "High-resolution renderer",
        f"Requests: {int(app['render_requests']):,}",
        f"Completed: {int(app['render_completed']):,}",
        f"Failed/interrupted: {int(app['render_failed']):,}",
        f"Cancelled: {int(app['render_cancelled']):,}",
        f"Rejected: {int(app['render_rejected']):,}",
        f"Compute time: {float(app['render_seconds']) / 60:.1f} minutes",
        "",
        "Operations",
        f"Backend starts: {int(app['backend_starts']):,}",
        f"Time studies: {int(app['time_studies_started']):,} started / "
        f"{int(app['time_studies_rejected']):,} rejected",
        f"Numerical failures: {int(app['numerical_failures']):,}",
        f"Internal errors: {int(app['internal_errors']):,}",
        f"Total recorded errors: {app_errors:,}",
        "",
        "This report contains daily aggregates only; no IPs, user agents, recipes,",
        "seeds, or visitor histories are stored by the application reporter.",
    ]
    return "\n".join(lines) + "\n"


def send_email(
    config: ReportSettings,
    day: date,
    body: str,
    *,
    subject: str | None = None,
) -> None:
    if not (config.report_to and config.report_from and config.smtp_host):
        raise ValueError(
            "TURING_REPORT_TO, TURING_REPORT_FROM, and TURING_SMTP_HOST are required"
        )
    message = EmailMessage()
    message["To"] = config.report_to
    message["From"] = config.report_from
    message["Subject"] = subject or f"Turing Pattern daily usage — {day.isoformat()}"
    message.set_content(body)
    context = ssl.create_default_context()
    if config.smtp_security == "ssl":
        client: smtplib.SMTP = smtplib.SMTP_SSL(
            config.smtp_host, config.smtp_port, timeout=30, context=context
        )
    else:
        client = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30)
    with client:
        if config.smtp_security == "starttls":
            client.starttls(context=context)
        if config.smtp_user:
            if config.smtp_password is None:
                raise ValueError(
                    "SMTP user is configured but its password file is missing"
                )
            client.login(config.smtp_user, config.smtp_password)
        client.send_message(message)


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send the daily aggregate usage report"
    )
    parser.add_argument("--date", type=date.fromisoformat, help="local report date")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="print without claiming or sending"
    )
    mode.add_argument(
        "--test-email",
        action="store_true",
        help="send a configuration test without claiming a report date",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    config = ReportSettings.from_env()
    zone = ZoneInfo(config.timezone_name)
    report_day = arguments.date or (datetime.now(zone).date() - timedelta(days=1))
    if arguments.test_email:
        send_email(
            config,
            report_day,
            "Turing Pattern reporting SMTP configuration succeeded.\n",
            subject="Turing Pattern reporting test",
        )
        print(f"Sent reporting test email to {config.report_to}")
        return 0
    store = UsageStore(config.data_dir, config.timezone_name)
    try:
        app = store.snapshot(report_day)
        try:
            edge = cloudflare_analytics(config, report_day)
        except Exception as error:  # noqa: BLE001 - application digest can still send
            edge = {"status": "unavailable", "detail": str(error)[:300]}
        body = build_digest(report_day, app, edge, config.hostname)
        if arguments.dry_run:
            print(body, end="")
            return 0
        if not store.claim_delivery(report_day):
            delivery = store.delivery(report_day)
            print(
                f"Report for {report_day.isoformat()} already claimed: "
                f"{delivery['status'] if delivery else 'unknown'}",
                file=sys.stderr,
            )
            return 0
        try:
            send_email(config, report_day, body)
        except Exception as error:
            store.finish_delivery(report_day, "failed", str(error))
            raise
        store.finish_delivery(report_day, "sent")
        print(f"Sent usage report for {report_day.isoformat()} to {config.report_to}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReportSettings",
    "build_digest",
    "cloudflare_analytics",
    "main",
    "send_email",
]
