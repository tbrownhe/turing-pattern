from datetime import UTC, date, datetime

import pytest

from app.usage import UsageStore


def test_usage_aggregates_persist_across_store_restarts(tmp_path):
    instant = datetime(2026, 7, 20, 6, 30, tzinfo=UTC)
    first = UsageStore(tmp_path, "America/Los_Angeles")
    first.increment("live_sessions_admitted", at=instant)
    first.increment_many({"live_frames": 12, "live_frame_bytes": 4096}, at=instant)
    first.record_peak("peak_compute_active", 2, at=instant)
    first.close()

    second = UsageStore(tmp_path, "America/Los_Angeles")
    snapshot = second.snapshot(date(2026, 7, 19))
    second.close()

    assert snapshot["live_sessions_admitted"] == 1
    assert snapshot["live_frames"] == 12
    assert snapshot["live_frame_bytes"] == 4096
    assert snapshot["peak_compute_active"] == 2


def test_usage_keeps_highest_compute_peak(tmp_path):
    store = UsageStore(tmp_path, "UTC")
    store.record_peak("peak_compute_active", 2)
    store.record_peak("peak_compute_active", 1)

    assert store.snapshot(datetime.now(UTC).date())["peak_compute_active"] == 2
    store.close()


def test_delivery_can_only_be_claimed_once_even_after_failure(tmp_path):
    report_day = date(2026, 7, 19)
    first = UsageStore(tmp_path, "UTC")
    second = UsageStore(tmp_path, "UTC")

    assert first.claim_delivery(report_day) is True
    assert second.claim_delivery(report_day) is False
    first.finish_delivery(report_day, "failed", "SMTP connection was ambiguous")
    assert second.claim_delivery(report_day) is False
    assert second.delivery(report_day)["status"] == "failed"

    first.close()
    second.close()


def test_usage_rejects_unknown_or_naive_inputs(tmp_path):
    store = UsageStore(tmp_path, "UTC")
    with pytest.raises(ValueError, match="unknown usage counters"):
        store.increment("visitor_ip")
    with pytest.raises(ValueError, match="timezone-aware"):
        store.increment("live_frames", at=datetime(2026, 7, 19))
    store.close()
