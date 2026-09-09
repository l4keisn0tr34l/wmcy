#!/usr/bin/env bash
set -euo pipefail

# Generate and immediately validate the controlled MVP corpus. This script must
# be run interactively because host tcpdump requires the owner's sudo approval.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLAN="${1:-$ROOT/configs/mvp_episode_plan.csv}"
PYTHON="$ROOT/.venv/bin/python"
PAUSE_FILE="$SCRIPT_DIR/.pause_corpus"

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
if [[ -e "$PAUSE_FILE" ]]; then
  echo "corpus generation is paused by $PAUSE_FILE" >&2
  echo "remove it to resume: rm -f $PAUSE_FILE" >&2
  exit 0
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
    duration = row.get("duration_seconds") or "120"
    if not duration.isdigit() or int(duration) < 120:
        raise SystemExit(f"invalid duration_seconds for {row['episode_id']}: {duration!r}")
    action = row.get("defender_action") or ""
    print(f"{row['episode_id']}\t{row['scenario']}\t{row['seed']}\t{duration}\t{action}")
PY
)

total="${#PLAN_ROWS[@]}"
completed=0
for row in "${PLAN_ROWS[@]}"; do
  # A sentinel requests a clean stop between episodes. Never suspend an active
  # scenario because wall-clock capture would continue and corrupt its timing.
  if [[ -e "$PAUSE_FILE" ]]; then
    echo
    echo "pause requested; stopped cleanly after $completed/$total plan rows"
    echo "resume with: rm -f $PAUSE_FILE && $0 $PLAN"
    exit 0
  fi
  IFS=$'\t' read -r episode_id scenario seed duration_seconds defender_action <<< "$row"
  completed=$((completed + 1))
  episode_dir="$SCRIPT_DIR/episodes/$episode_id"
  echo
  echo "===== MVP episode $completed/$total: $episode_id ====="

  if [[ -f "$episode_dir/network.pcap" && -f "$episode_dir/ground_truth.csv" && -f "$episode_dir/episode_metadata.csv" ]]; then
    if [[ -n "$defender_action" && ! -f "$episode_dir/defender_actions.csv" ]]; then
      echo "cannot resume: plan requires missing $episode_dir/defender_actions.csv" >&2
      exit 1
    fi
    # Resume only when existing raw data exactly matches this plan row.
    "$PYTHON" - "$episode_dir/episode_metadata.csv" "$episode_id" "$scenario" "$seed" "$duration_seconds" "$defender_action" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 1:
    raise SystemExit(f"cannot resume: invalid metadata row count in {sys.argv[1]}")
expected = {"episode_id": sys.argv[2], "scenario": sys.argv[3], "seed": sys.argv[4]}
actual = {key: rows[0].get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"cannot resume: metadata {actual} does not match plan {expected}")
planned_duration = rows[0].get("planned_capture_duration_seconds")
if planned_duration is not None and planned_duration != sys.argv[5]:
    raise SystemExit(
        f"cannot resume: metadata duration {planned_duration!r} does not match plan {sys.argv[5]!r}"
    )
if planned_duration is None:
    print("resume: legacy metadata has no planned duration; base fields match")
expected_action = sys.argv[6]
if expected_action and rows[0].get("defender_action") != expected_action:
    raise SystemExit(
        f"cannot resume: defender action {rows[0].get('defender_action')!r} != {expected_action!r}"
    )
PY
    if [[ -f "$episode_dir/observations.csv.gz" && -d "$episode_dir/states" && -f "$episode_dir/state_ground_truth.csv" ]] \
       && "$PYTHON" "$ROOT/scripts/07_validate_episode.py" "$episode_dir" --window-seconds 5; then
      echo "resume: existing episode already passes; skipping capture"
      continue
    fi
    echo "resume: raw episode exists; rebuilding derived files without recapture"
    "$PYTHON" "$ROOT/scripts/08_process_lab_episode.py" "$episode_dir" --force
    continue
  fi

  if [[ -e "$episode_dir" ]]; then
    echo "cannot resume partial raw episode directory: $episode_dir" >&2
    exit 1
  fi
  "$SCRIPT_DIR/run_episode.sh" "$episode_id" --scenario "$scenario" --seed "$seed" \
    --duration-seconds "$duration_seconds"
  "$PYTHON" "$ROOT/scripts/08_process_lab_episode.py" "$episode_dir"
done

echo
echo "MVP corpus generation complete: $total/$total planned episodes passed processing and validation."
