#!/usr/bin/env bash
set -euo pipefail

# Generate and immediately validate the controlled MVP corpus. This script must
# be run interactively because host tcpdump requires the owner's sudo approval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLAN="${1:-$ROOT/configs/mvp_episode_plan.csv}"
PYTHON="$ROOT/.venv/bin/python"

if [[ ! -f "$PLAN" ]]; then
  echo "missing episode plan: $PLAN" >&2
  exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "missing project Python: $PYTHON" >&2
  exit 1
fi
if [[ "$(docker compose -f "$SCRIPT_DIR/docker-compose.yml" ps --services --status running | wc -l)" -ne 3 ]]; then
  echo "all three lab containers must be running" >&2
  exit 1
fi

# Authenticate once, then refresh only the timestamp while this foreground corpus
# job is alive. No command other than tcpdump/kill is run with sudo by episodes.
sudo -v
while true; do
  sudo -n true 2>/dev/null || exit
  sleep 60
  kill -0 "$$" 2>/dev/null || exit
done &
SUDO_KEEPALIVE_PID=$!
cleanup() {
  kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true
}
trap cleanup EXIT

mapfile -t PLAN_ROWS < <("$PYTHON" - "$PLAN" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
required = {"episode_id", "scenario", "seed"}
if not rows or not required.issubset(rows[0]):
    raise SystemExit("plan must contain episode_id,scenario,seed and at least one row")
for row in rows:
    print(f"{row['episode_id']}\t{row['scenario']}\t{row['seed']}")
PY
)

total="${#PLAN_ROWS[@]}"
completed=0
for row in "${PLAN_ROWS[@]}"; do
  IFS=$'\t' read -r episode_id scenario seed <<< "$row"
  completed=$((completed + 1))
  echo
  echo "===== MVP episode $completed/$total: $episode_id ====="
  "$SCRIPT_DIR/run_episode.sh" "$episode_id" --scenario "$scenario" --seed "$seed"
  "$PYTHON" "$ROOT/scripts/08_process_lab_episode.py" "$SCRIPT_DIR/episodes/$episode_id"
done

echo
echo "MVP corpus generation complete: $total/$total planned episodes passed processing and validation."
