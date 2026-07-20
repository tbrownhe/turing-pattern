# Privacy

The Turing Pattern application accepts simulation controls and seeds only to produce
the requested live or rendered output. The daily reporting subsystem stores daily
aggregate counters: session and render counts, duration, frame and artifact transfer,
peak compute use, failures, and backend starts. It does not store IP addresses, user
agents, recipes, seeds, or per-visitor histories.

Estimated visits, HTTP requests, and edge transfer in the email are queried as
hostname-level aggregates from Cloudflare. The application does not import or retain
Cloudflare request records. Cloudflare and the deployment proxy may independently
process connection metadata according to their own configuration and policies.

Completed render artifacts and their embedded recipes remain temporarily available
under the configured artifact retention policy so the requesting browser can download
them. This operational storage is separate from daily reporting.

Use the repository's GitHub Issues for non-sensitive questions and problem reports.
Report security-sensitive concerns privately through the process in
[SECURITY.md](../.github/SECURITY.md).
