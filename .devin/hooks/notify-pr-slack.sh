#!/usr/bin/env bash
set -euo pipefail

hook_input=$(cat)

HOOK_INPUT=$hook_input python3 - <<'PY' || true
import hashlib
import json
import os
import re
import shlex
import sys
import tempfile
import urllib.parse
import urllib.request

SHELL_SEPARATORS = {";", "&&", "||", "|", "&", "\n", "(", ")", "{", "}"}
GH_GLOBAL_FLAGS_WITH_VALUE = {"-R", "--repo"}
COMMAND_WRAPPERS = {"command", "env", "exec", "nohup", "time", "sudo"}


def split_segments(command):
    """Split a shell command line into the argv of each simple command."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|(){}")
    lexer.whitespace_split = True
    lexer.whitespace = " \t\r"
    lexer.commenters = ""
    segments, current = [], []
    for token in lexer:
        if token in SHELL_SEPARATORS or all(c in ";&|" for c in token):
            if current:
                segments.append(current)
            current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def gh_pr_create_args(argv):
    """Return the args after `gh [global flags] pr create`, or None."""
    i = 0
    while i < len(argv) and (
        argv[i] in COMMAND_WRAPPERS or re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[i])
    ):
        i += 1
    if i >= len(argv) or os.path.basename(argv[i]) != "gh":
        return None
    i += 1
    while i < len(argv) and argv[i].startswith("-"):
        if argv[i] in GH_GLOBAL_FLAGS_WITH_VALUE:
            i += 2
        else:
            i += 1
    if argv[i : i + 2] != ["pr", "create"]:
        return None
    return argv[i + 2 :]


def extract_title(args):
    for j, arg in enumerate(args):
        if arg in ("-t", "--title") and j + 1 < len(args):
            return args[j + 1]
        if arg.startswith("--title="):
            return arg[len("--title=") :]
        if arg.startswith("-t") and len(arg) > 2 and not arg.startswith("--"):
            return arg[2:]
    return None


try:
    payload = json.loads(os.environ["HOOK_INPUT"])

    debug_log = os.environ.get("DEVIN_PR_HOOK_DEBUG_LOG", "").strip()
    if debug_log:
        try:
            fd = os.open(debug_log, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        sys.exit(0)
    parsed = urllib.parse.urlparse(webhook)
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
    ):
        print("notify-pr-slack: SLACK_WEBHOOK_URL must use https", file=sys.stderr)
        sys.exit(0)

    command = (payload.get("tool_input") or {}).get("command") or ""
    try:
        segments = split_segments(command)
    except ValueError:
        sys.exit(0)
    create_args = None
    for argv in segments:
        create_args = gh_pr_create_args(argv)
        if create_args is not None:
            break
    if create_args is None:
        sys.exit(0)

    response = payload.get("tool_response") or {}
    if not response.get("success"):
        sys.exit(0)

    output = response.get("output") or ""
    m = re.search(r"https://github\.com/[^\s]+/pull/\d+", output)
    if not m:
        sys.exit(0)
    pr_ref = m.group(0)

    marker_dir = os.path.join(tempfile.gettempdir(), "devin-pr-slack-notified")
    os.makedirs(marker_dir, exist_ok=True)
    marker = os.path.join(marker_dir, hashlib.sha1(pr_ref.encode()).hexdigest())
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        sys.exit(0)

    try:
        project_dir = os.environ.get("DEVIN_PROJECT_DIR") or os.getcwd()
        repo = os.path.basename(project_dir.rstrip("/")) or "repo"

        text = f":rocket: Devin CLI opened a PR in {repo}: {pr_ref}"
        title = extract_title(create_args)
        if title:
            text += f" — {title}"

        body = json.dumps({"text": text}).encode()
        req = urllib.request.Request(
            webhook, data=body, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        try:
            os.unlink(marker)
        except OSError:
            pass
        raise
except Exception as exc:
    print(f"notify-pr-slack: {exc}", file=sys.stderr)
    sys.exit(0)
PY
