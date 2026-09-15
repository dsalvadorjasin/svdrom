#!/usr/bin/env bash
set -euo pipefail

hook_input=$(cat)

HOOK_INPUT=$hook_input python3 - <<'PY' || true
import json
import os
import re
import sys
import urllib.request

try:
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if not webhook:
        sys.exit(0)

    payload = json.loads(os.environ["HOOK_INPUT"])

    command = (payload.get("tool_input") or {}).get("command") or ""
    if not re.search(r"\bgh\s+pr\s+create\b", command):
        sys.exit(0)

    response = payload.get("tool_response") or {}
    if not response.get("success"):
        sys.exit(0)

    output = response.get("output") or ""
    m = re.search(r"https://github\.com/[^\s]+/pull/\d+", output)
    pr_ref = m.group(0) if m else "PR opened (URL not found in output)"

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
