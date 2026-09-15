#!/usr/bin/env bash
set -euo pipefail

hook_input=$(cat)

HOOK_INPUT=$hook_input python3 - <<'PY' || true
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request

try:
    payload = json.loads(os.environ["HOOK_INPUT"])

    debug_log = os.environ.get("DEVIN_PR_HOOK_DEBUG_LOG", "").strip()
    if debug_log:
        try:
            with open(debug_log, "a") as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        sys.exit(0)

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bgh\s+pr\s+create\b", command):
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

    project_dir = os.environ.get("DEVIN_PROJECT_DIR") or os.getcwd()
    repo = os.path.basename(project_dir.rstrip("/")) or "repo"

    text = f":rocket: Devin CLI opened a PR in {repo}: {pr_ref}"
    tm = re.search(r'--title\s+"([^"]+)"|--title\s+\'([^\']+)\'', command)
    if tm:
        title = tm.group(1) or tm.group(2)
        text += f" — {title}"

    body = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook, data=body, headers={"Content-Type": "application/json"}
    )
    urllib.request.urlopen(req, timeout=10)
except Exception as exc:
    print(f"notify-pr-slack: {exc}", file=sys.stderr)
    sys.exit(0)
PY
