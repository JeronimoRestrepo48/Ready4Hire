"""Pytest fixtures shared across the Ready4Hire test suite."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from typing import Generator

import httpx
import pytest


@pytest.fixture(scope="session", autouse=True)
def backend_server() -> Generator[None, None, None]:
    """Launch the FastAPI backend on port 8001 for integration tests.

    The server is started once per test session and shut down at the end. The
    fixture enables MOCK_OLLAMA so that tests do not depend on an external LLM
    instance.
    """

    env = os.environ.copy()
    env.setdefault("MOCK_OLLAMA", "true")
    env.setdefault("PYTHONUNBUFFERED", "1")

    command = [
        "uvicorn",
        "app.main_v2_improved:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8001",
    ]

    process = subprocess.Popen(  # noqa: S603, S607 - controlled input
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        env=env,
    )

    try:
        _wait_for_startup()
        yield
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def _wait_for_startup() -> None:
    """Poll the health endpoint until the backend reports healthy."""

    base_url = "http://127.0.0.1:8001/api/v2/health"
    timeout_seconds = 60
    poll_interval = 1.0
    deadline = time.monotonic() + timeout_seconds

    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(base_url)
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except Exception as exc:  # noqa: BLE001 - we want to keep retrying
            last_error = str(exc)
        time.sleep(poll_interval)

    raise RuntimeError(
        "FastAPI backend did not become ready within the allotted time. "
        f"Last error: {last_error}"
    )
