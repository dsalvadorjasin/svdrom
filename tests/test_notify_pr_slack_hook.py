"""Tests for the Devin CLI hook in .devin/hooks/notify-pr-slack.sh."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[1] / ".devin" / "hooks" / "notify-pr-slack.sh"
PR_URL = "https://github.com/acme/widgets/pull/42"


class _Receiver:
    def __init__(self) -> None:
        self.posts: list[dict[str, str]] = []
        self.fail_next = False
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers["Content-Length"])
                receiver.posts.append(json.loads(self.rfile.read(length)))
                self.send_response(503 if receiver.fail_next else 200)
                receiver.fail_next = False
                self.end_headers()

            def log_message(self, *_: object) -> None:
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/hook"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture()
def receiver() -> Iterator[_Receiver]:
    r = _Receiver()
    yield r
    r.server.shutdown()
    r.server.server_close()


@pytest.fixture()
def run_hook(receiver: _Receiver, tmp_path: Path):
    def _run(
        command: str,
        output: str = PR_URL,
        success: bool = True,
        webhook: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        payload = {
            "tool_input": {"command": command},
            "tool_response": {"output": output, "success": success},
        }
        env = {k: v for k, v in os.environ.items() if k != "SLACK_WEBHOOK_URL"}
        env["TMPDIR"] = str(tmp_path)
        if webhook is None:
            env["SLACK_WEBHOOK_URL"] = receiver.url
        elif webhook:
            env["SLACK_WEBHOOK_URL"] = webhook
        result = subprocess.run(
            [str(HOOK)],
            input=json.dumps(payload),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        assert result.stdout == ""
        return result

    return _run


@pytest.mark.parametrize(
    ("command", "title"),
    [
        ('gh pr create --title "Hello world" --body x', "Hello world"),
        ("gh -R acme/widgets pr create --title=Fix", "Fix"),
        ("git push -u origin x && gh pr create -t Chained --fill", "Chained"),
        ("cd repo; GH_TOKEN=x gh pr create --fill", None),
    ],
)
def test_posts_for_gh_pr_create(run_hook, receiver, command, title) -> None:
    run_hook(command)
    assert len(receiver.posts) == 1
    text = receiver.posts[0]["text"]
    assert PR_URL in text
    assert (title in text) if title else ("\u2014" not in text)


@pytest.mark.parametrize(
    "command",
    [
        "printf 'gh pr create\\nhttps://github.com/acme/widgets/pull/42\\n'",
        'echo "gh pr create"',
        "gh pr view 42",
        "gh pr create-ish",
        "ls",
    ],
)
def test_ignores_non_creating_commands(run_hook, receiver, command) -> None:
    run_hook(command)
    assert receiver.posts == []


def test_ignores_failed_or_urlless_commands(run_hook, receiver) -> None:
    run_hook("gh pr create", success=False)
    run_hook("gh pr create", output="no url here")
    assert receiver.posts == []


def test_deduplicates_per_pr(run_hook, receiver) -> None:
    run_hook("gh pr create", output="no url yet")
    run_hook("gh pr create")
    run_hook("gh pr create")
    assert len(receiver.posts) == 1


def test_retries_after_delivery_failure(run_hook, receiver) -> None:
    receiver.fail_next = True
    result = run_hook("gh pr create")
    assert "503" in result.stderr
    run_hook("gh pr create")
    assert len(receiver.posts) == 2


def test_noop_without_webhook_or_with_insecure_webhook(run_hook, receiver) -> None:
    run_hook("gh pr create", webhook="")
    result = run_hook("gh pr create", webhook="http://example.com/hook")
    assert "https" in result.stderr
    assert receiver.posts == []
