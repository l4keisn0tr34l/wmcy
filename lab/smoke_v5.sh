#!/usr/bin/env bash
# Exactly two smoke captures; deliberately does NOT freeze or start the corpus.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="$ROOT/.venv/bin/python"
if [[ "${V5_SYSTEM_INHIBITED:-0}" != 1 ]]; then
  command -v systemd-inhibit >/dev/null || { echo "systemd-inhibit is required" >&2; exit 1; }
  exec systemd-inhibit --what=shutdown:sleep:idle --mode=block \
    --who=cyberwm-v5-smoke --why="Protect complete V5 capture windows" \
    env V5_SYSTEM_INHIBITED=1 bash "$0" "$@"
fi
cd "$ROOT"
exec 9>"$SCRIPT_DIR/.v5_corpus.lock"
flock -n 9 || { echo "another capture controller is running" >&2; exit 1; }
[[ ! -e "$ROOT/configs/mvp_v5_capture_freeze.json" ]] || { echo "already frozen; do not recapture smokes" >&2; exit 1; }
"$PYTHON" scripts/46_validate_v5_plan.py
"$PYTHON" scripts/48_test_v5_contract.py
sudo -v
while true; do sudo -n true 2>/dev/null || exit; sleep 60; kill -0 "$$" 2>/dev/null || exit; done &
KEEPALIVE=$!
trap 'kill "$KEEPALIVE" 2>/dev/null || true' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
for row in 'v5_smoke_001 legitimate_ssh 90001' 'v5_smoke_002 scan_guess_action_block 90002'; do
  read -r id scenario seed <<< "$row"
  episode="$SCRIPT_DIR/episodes/$id"
  if [[ ! -e "$episode" ]]; then
    "$SCRIPT_DIR/run_v5_episode.sh" "$id" --scenario "$scenario" --seed "$seed" --background mixed
  fi
  [[ -f "$episode/episode_metadata.csv" && -f "$episode/cleanup.json" ]] || {
    echo "partial smoke needs quarantine, not overwrite: $episode" >&2; exit 1;
  }
  if [[ ! -e "$episode/states" && ! -e "$episode/observations.csv.gz" && ! -e "$episode/state_ground_truth.csv" ]]; then
    "$PYTHON" scripts/08_process_lab_episode.py "$episode"
  fi
  "$PYTHON" scripts/07_validate_episode.py "$episode" --window-seconds 5
  "$PYTHON" scripts/49_freeze_v5_capture.py --episode "$episode"
done
# Derived exporters refuse overwrite; resumed smokes are validated above.
for mode in passive action passive_action; do
  if [[ ! -e "$ROOT/outputs/mvp_v5/sequences/$mode/smoke" ]]; then
    "$PYTHON" scripts/47_build_v5_sequences.py --split smoke --mode "$mode"
  fi
done
echo "Two smokes complete. Review artifacts, then explicitly freeze. No corpus started."
