# Daily usage report

The optional reporter emails one plain-text summary for the preceding complete local
day. It combines privacy-safe application counters from the existing `render-data`
volume with aggregated Cloudflare hostname analytics. Reporting never runs in a web
request and does not need the frontend or backend containers to be running.

## Configure it

Create a custom Cloudflare API token with **Account > Account Analytics > Read** and
limit its zone resources to the zone containing `tobiasbrownheft.xyz`, then copy the
zone ID from the domain overview. Configure a generic SMTP relay; Gmail SMTP works
with an app password when two-step verification is enabled, but the reporter is not
tied to Gmail.

References: [Cloudflare Analytics token configuration](https://developers.cloudflare.com/analytics/graphql-api/getting-started/authentication/api-token-auth/)
and [Google's SMTP settings](https://support.google.com/a/answer/176600).

Add the non-secret values to `.env`:

```dotenv
TURING_REPORT_TIMEZONE=America/Los_Angeles
TURING_REPORT_TO=tbrownhe@gmail.com
TURING_REPORT_FROM=tbrownhe@gmail.com
TURING_SMTP_HOST=smtp.gmail.com
TURING_SMTP_PORT=587
TURING_SMTP_SECURITY=starttls
TURING_SMTP_USER=tbrownhe@gmail.com
TURING_CLOUDFLARE_ZONE_ID=replace-with-zone-id
TURING_REPORT_SECRETS_DIR=./secrets/reporting
```

Put the two secrets in files. Do not add a trailing label or quote around either
value:

```console
mkdir -p secrets/reporting
printf '%s' 'replace-with-cloudflare-token' > secrets/reporting/cloudflare-token
printf '%s' 'replace-with-smtp-password' > secrets/reporting/smtp-password
sudo chgrp -R 10001 secrets secrets/reporting/*
chmod 750 secrets secrets/reporting
chmod 640 secrets/reporting/*
```

The backend image runs as UID/GID 10001, so the group permissions let the reporter
read the files while keeping them private from other host users. The whole `secrets/`
directory is ignored by Git. Back up these two credentials in your password manager;
the files themselves are disposable and can be recreated.

## Test before sending

Build the production image, test SMTP without claiming a date, then preview
yesterday's digest. A dry run neither sends mail nor claims the date:

```console
docker compose build backend
docker compose --profile reporting run --rm reporter --test-email
docker compose --profile reporting run --rm reporter --dry-run
```

To preview a particular local day:

```console
docker compose --profile reporting run --rm reporter --dry-run --date 2026-07-19
```

Send the real report once configuration looks right:

```console
docker compose --profile reporting run --rm reporter
```

The reporter records a delivery claim in `/var/lib/turing/usage.sqlite3` before it
contacts SMTP. Concurrent or repeated invocations therefore cannot duplicate a
message. This is strict at-most-once delivery: if SMTP fails after the claim, the
failure is recorded and automatic retries for that date are refused because the
remote server may already have accepted the email. Use `--dry-run --date ...` to
inspect that day's data without altering its delivery record.

Cloudflare errors do not suppress the application report; the email says analytics
were unavailable. SMTP configuration errors do fail the one-shot job.

## Schedule on silicide

The repository includes a systemd one-shot service and timer configured for
`/srv/turing`. Install and enable them:

```console
sudo cp deploy/systemd/turing-report.service /etc/systemd/system/
sudo cp deploy/systemd/turing-report.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now turing-report.timer
systemctl list-timers turing-report.timer
```

It runs at 08:15 local server time with a randomized delay and catches up after a
reboot. Inspect the most recent attempt with:

```console
systemctl status turing-report.service
journalctl -u turing-report.service -n 100 --no-pager
```

The scheduler is intentionally separate from outage alerting.
