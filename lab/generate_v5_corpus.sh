#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
PLAN="${1:-$ROOT/configs/mvp_v5_episode_plan.csv}"
PAUSE_FILE="$SCRIPT_DIR/.pause_v5_corpus"
[[ "$(realpath "$PLAN")" == "$ROOT/configs/mvp_v5_episode_plan.csv" ]] || { echo "only the frozen V5 plan is accepted" >&2; exit 1; }
exec 9>"$SCRIPT_DIR/.v5_corpus.lock"
flock -n 9 || { echo "another corpus controller is running" >&2; exit 1; }
"$PYTHON" "$ROOT/scripts/46_validate_v5_plan.py"
"$PYTHON" "$ROOT/scripts/49_freeze_v5_capture.py" --check
[[ ! -e "$PAUSE_FILE" ]] || { echo "V5 paused by $PAUSE_FILE"; exit 0; }
sudo -v
while true; do sudo -n true 2>/dev/null || exit; sleep 60; kill -0 "$$" 2>/dev/null || exit; done &
KEEPALIVE=$!
trap 'kill "$KEEPALIVE" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
mapfile -t ROWS < <("$PYTHON" - "$PLAN" "$ROOT/configs/mvp_v5_capture_freeze.json" <<'PY'
import csv, json, sys
with open(sys.argv[1]) as f: rows = {r['episode_id']: r for r in csv.DictReader(f)}
with open(sys.argv[2]) as f: order = json.load(f)['capture_order']
assert set(order) == set(rows) and len(order) == len(rows)
for name in order:
    print('\t'.join(rows[name][key] for key in ['episode_id','scenario','seed','duration_seconds','background_profile']))
PY
)
[[ "${#ROWS[@]}" -eq 80 ]] || { echo "invalid frozen capture order" >&2; exit 1; }
completed=0
for row in "${ROWS[@]}"; do
  [[ ! -e "$PAUSE_FILE" ]] || { echo "paused after $completed/80; resume after removing $PAUSE_FILE"; exit 0; }
  IFS=$'\t' read -r episode_id scenario seed duration background <<< "$row"
  episode_dir="$SCRIPT_DIR/episodes/$episode_id"
  echo "===== V5 $((completed+1))/80: $episode_id ====="
  if [[ ! -e "$episode_dir" ]]; then
    "$SCRIPT_DIR/run_v5_episode.sh" "$episode_id" --scenario "$scenario" --seed "$seed" \
      --duration-seconds "$duration" --background "$background"
  fi
  [[ -f "$episode_dir/episode_metadata.csv" && -f "$episode_dir/cleanup.json" ]] || {
    echo "partial raw capture must be quarantined, never resumed: $episode_dir" >&2; exit 1;
  }
  if [[ ! -e "$episode_dir/observations.csv.gz" && ! -e "$episode_dir/states" && ! -e "$episode_dir/state_ground_truth.csv" ]]; then
    "$PYTHON" "$ROOT/scripts/08_process_lab_episode.py" "$episode_dir"
  fi
  "$PYTHON" "$ROOT/scripts/07_validate_episode.py" "$episode_dir" --window-seconds 5
  "$PYTHON" "$ROOT/scripts/49_freeze_v5_capture.py" --episode "$episode_dir"
  completed=$((completed+1))
done
echo "V5 complete: $completed/80 captured and structurally validated. Test model evaluation remains sealed."
