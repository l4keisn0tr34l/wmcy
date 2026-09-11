#!/usr/bin/env bash
set -euo pipefail

# Resumable V5 capture/processing. Pause only between episodes by creating
# lab/.pause_v5_corpus; never suspend an active capture.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PLAN="${1:-$ROOT/configs/mvp_v5_episode_plan.csv}"
PYTHON="$ROOT/.venv/bin/python"
PAUSE_FILE="$SCRIPT_DIR/.pause_v5_corpus"
COMPOSE=(docker compose -p cyberwm_v5 -f "$SCRIPT_DIR/docker-compose-v5.yml")

[[ -f "$PLAN" ]] || { echo "missing V5 plan: $PLAN" >&2; exit 1; }
[[ -x "$PYTHON" ]] || { echo "missing project Python" >&2; exit 1; }
"$PYTHON" "$ROOT/scripts/46_validate_v5_plan.py" --plan "$PLAN"
[[ "$("${COMPOSE[@]}" ps --services --status running | wc -l)" -eq 5 ]] || {
  echo "all five V5 containers must be running" >&2; exit 1;
}
[[ ! -e "$PAUSE_FILE" ]] || { echo "V5 capture paused by $PAUSE_FILE"; exit 0; }

sudo -v
while true; do sudo -n true 2>/dev/null || exit; sleep 60; kill -0 "$$" 2>/dev/null || exit; done &
SUDO_KEEPALIVE_PID=$!
cleanup() { kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true; }
trap cleanup EXIT

mapfile -t ROWS < <("$PYTHON" - "$PLAN" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
for row in rows:
    print("\t".join(row[key] for key in ["episode_id", "scenario", "seed", "duration_seconds", "defender_action", "background_profile", "topology_profile", "node_count"]))
PY
)

total="${#ROWS[@]}"; completed=0
for row in "${ROWS[@]}"; do
  if [[ -e "$PAUSE_FILE" ]]; then
    echo "pause requested after $completed/$total episodes"
    echo "resume: rm -f '$PAUSE_FILE' && '$0' '$PLAN'"
    exit 0
  fi
  IFS=$'\t' read -r episode_id scenario seed duration action background topology node_count <<< "$row"
  completed=$((completed + 1)); episode_dir="$SCRIPT_DIR/episodes/$episode_id"
  echo; echo "===== V5 $completed/$total: $episode_id ====="
  if [[ -f "$episode_dir/network.pcap" && -f "$episode_dir/ground_truth.csv" && -f "$episode_dir/episode_metadata.csv" && -f "$episode_dir/defender_actions.csv" ]]; then
    "$PYTHON" - "$episode_dir/episode_metadata.csv" "$episode_id" "$scenario" "$seed" "$duration" "$action" "$background" "$topology" "$node_count" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
if len(rows) != 1: raise SystemExit("cannot resume invalid metadata")
keys = ["episode_id", "scenario", "seed", "planned_capture_duration_seconds", "defender_action", "background_profile", "topology_profile", "node_count"]
expected = dict(zip(keys, sys.argv[2:]))
actual = {key: rows[0].get(key) for key in keys}
if actual != expected: raise SystemExit(f"cannot resume metadata {actual} != plan {expected}")
PY
    if [[ -f "$episode_dir/observations.csv.gz" && -d "$episode_dir/states" && -f "$episode_dir/state_ground_truth.csv" ]] \
       && "$PYTHON" "$ROOT/scripts/07_validate_episode.py" "$episode_dir" --window-seconds 5; then
      echo "resume: existing V5 episode passes; skipping"
      continue
    fi
    echo "resume: rebuilding derived files from complete raw episode"
    "$PYTHON" "$ROOT/scripts/08_process_lab_episode.py" "$episode_dir" --force
    continue
  fi
  [[ ! -e "$episode_dir" ]] || { echo "partial episode must be quarantined: $episode_dir" >&2; exit 1; }
  "$SCRIPT_DIR/run_v5_episode.sh" "$episode_id" --scenario "$scenario" --seed "$seed" \
    --duration-seconds "$duration" --background "$background"
  "$PYTHON" "$ROOT/scripts/08_process_lab_episode.py" "$episode_dir"
done

echo; echo "V5 corpus complete: $total/$total episodes captured, processed, and validated."
