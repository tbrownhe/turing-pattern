"""Run every local correctness gate used by CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
UV = [
    "uv",
    "run",
    "--with-requirements",
    "requirements-dev.lock",
]
SCRIPT_FILES = [
    "../scripts/check.py",
    "../scripts/engine_benchmark.py",
    "../scripts/load_test.py",
    "../scripts/smoke_test.py",
]


def run(command: list[str], cwd: Path) -> None:
    print(f"\n> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    run([*UV, "ruff", "check", "app", "tests", *SCRIPT_FILES], BACKEND)
    run(
        [*UV, "ruff", "format", "--check", "app", "tests", *SCRIPT_FILES],
        BACKEND,
    )
    run([*UV, "mypy", "app"], BACKEND)
    run([*UV, "python", "-m", "pytest", "-q"], BACKEND)

    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    npx = "npx.cmd" if sys.platform == "win32" else "npx"
    run([npm, "ci"], FRONTEND)
    run([npm, "audit", "--audit-level=high"], FRONTEND)
    run([npm, "test"], FRONTEND)
    run([npm, "run", "lint", "--", "--max-warnings=0"], FRONTEND)
    run([npm, "run", "build"], FRONTEND)
    run([npx, "playwright", "install", "chromium"], FRONTEND)
    run([npm, "run", "test:e2e"], FRONTEND)


if __name__ == "__main__":
    main()
