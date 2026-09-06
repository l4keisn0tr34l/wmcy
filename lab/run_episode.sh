#!/usr/bin/env bash
set -euo pipefail

# Controlled ATT&CK-labelled episodes for the private Docker lab (10.77.0.0/24).
# Scenario metadata and ATT&CK truth are written separately from packet evidence.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE=(docker compose -f "$SCRIPT_DIR/docker-compose.yml")

usage() {
  cat <<'EOF'
Usage: ./run_episode.sh EPISODE_ID [options]

Options:
  --scenario NAME   benign_ping | legitimate_ssh | scan_only |
                    failed_guessing | one_hop | two_hop (default: two_hop)
  --seed INTEGER    Seed for host-role and timing randomization (default: epoch time)
  --out-dir PATH    Output directory (default: lab/episodes/EPISODE_ID)
  --help            Show this help

Use opaque episode IDs such as lab_002. Do not encode the scenario name in an ID
that may later appear in model-facing dataset metadata.
EOF
}

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

EPISODE_ID="$1"
shift
SCENARIO="two_hop"
SEED="$(date +%s)"
OUT_DIR="$SCRIPT_DIR/episodes/$EPISODE_ID"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO="${2:?missing --scenario value}"; shift 2 ;;
    --seed) SEED="${2:?missing --seed value}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?missing --out-dir value}"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "$SCENARIO" in
  benign_ping|legitimate_ssh|scan_only|failed_guessing|one_hop|two_hop) ;;
  *) echo "invalid scenario: $SCENARIO" >&2; usage >&2; exit 2 ;;
esac
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "seed must be a non-negative integer" >&2
  exit 2
fi
if [[ ! "$EPISODE_ID" =~ ^[A-Za-z0-9_.-]+$ ]]; then
  echo "episode ID may contain only letters, numbers, dot, underscore, and hyphen" >&2
  exit 2
fi

GT="$OUT_DIR/ground_truth.csv"
PCAP="$OUT_DIR/network.pcap"
META="$OUT_DIR/episode_metadata.csv"
if [[ -e "$PCAP" || -e "$GT" || -e "$META" ]]; then
  echo "refusing to overwrite existing raw episode files in $OUT_DIR" >&2
  exit 1
fi

# Use a literal decimal point rather than the locale-dependent decimal separator
# used by `date --iso-8601=ns`; a comma would corrupt the CSV column structure.
iso_now() { date '+%Y-%m-%dT%H:%M:%S.%N%:z'; }
record_event() {
  local start="$1" end="$2" actor="$3" target="$4" tid="$5" tech="$6" tactic="$7"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$EPISODE_ID" "$start" "$end" "$actor" "$target" "$tid" "$tech" "$tactic" >> "$GT"
}

host_ip() {
  case "$1" in
    ws1) echo '10.77.0.20' ;;
    srv1) echo '10.77.0.30' ;;
    srv2) echo '10.77.0.40' ;;
    *) echo "unknown lab host: $1" >&2; return 1 ;;
  esac
}

# Generate one deterministic host permutation from the recorded seed.
mapfile -t ROLE_HOSTS < <(python3 - "$SEED" <<'PY'
import random
import sys
hosts = ["ws1", "srv1", "srv2"]
random.Random(int(sys.argv[1])).shuffle(hosts)
print(*hosts, sep="\n")
PY
)
ACTOR="${ROLE_HOSTS[0]}"
PIVOT="${ROLE_HOSTS[1]}"
TARGET="${ROLE_HOSTS[2]}"
ACTOR_IP="$(host_ip "$ACTOR")"
PIVOT_IP="$(host_ip "$PIVOT")"
TARGET_IP="$(host_ip "$TARGET")"

# Bash's PRNG controls only timing/attempt-count variation; the seed is recorded.
RANDOM=$((SEED % 32768))
random_delay() {
  local minimum="$1" maximum="$2"
  echo $((minimum + RANDOM % (maximum - minimum + 1)))
}

compose_exec() {
  local host="$1"
  shift
  "${COMPOSE[@]}" exec -T "$host" "$@"
}

run_discovery() {
  local start end
  echo "[action] T1046 discovery from $ACTOR"
  start="$(iso_now)"
  compose_exec "$ACTOR" nmap -sT -Pn -p 22,80,443,445,3389 10.77.0.0/24 >/dev/null
  end="$(iso_now)"
  record_event "$start" "$end" "$ACTOR" internal_subnet T1046 'Network Service Discovery' Discovery
}

run_guessing() {
  local attempts start end pw
  attempts="$(random_delay 3 6)"
  echo "[action] T1110.001 $attempts failed passwords: $ACTOR -> $PIVOT"
  start="$(iso_now)"
  for ((i=1; i<=attempts; i++)); do
    pw="wrong${i}"
    compose_exec "$ACTOR" bash -lc \
      "sshpass -p '$pw' ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=2 lab@$PIVOT_IP 'true'" \
      >/dev/null 2>&1 || true
  done
  end="$(iso_now)"
  record_event "$start" "$end" "$ACTOR" "$PIVOT" T1110.001 'Password Guessing' 'Credential Access'
}

run_attack_ssh() {
  local source="$1" destination="$2" destination_ip="$3" start end
  echo "[action] T1021.004 SSH lateral movement: $source -> $destination"
  start="$(iso_now)"
  compose_exec "$source" bash -lc \
    "sshpass -p labpass ssh -o StrictHostKeyChecking=no lab@$destination_ip 'hostname && id'" >/dev/null
  end="$(iso_now)"
  record_event "$start" "$end" "$source" "$destination" T1021.004 SSH 'Lateral Movement'
}

run_legitimate_ssh() {
  echo "[action] legitimate administrative SSH: $ACTOR -> $PIVOT (no ATT&CK truth)"
  compose_exec "$ACTOR" bash -lc \
    "sshpass -p labpass ssh -o StrictHostKeyChecking=no lab@$PIVOT_IP 'hostname && uptime'" >/dev/null
}

cleanup() {
  if [[ -n "${TCPDUMP_PID:-}" ]]; then
    sudo -n kill "$TCPDUMP_PID" 2>/dev/null || true
    wait "$TCPDUMP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

running_services="$("${COMPOSE[@]}" ps --services --status running | wc -l)"
if [[ "$running_services" -ne 3 ]]; then
  echo "all three lab services must be running; use: cd $SCRIPT_DIR && docker compose up -d --build" >&2
  exit 1
fi

# Ask for authorization before creating files or starting background capture, so
# sudo cannot block invisibly and failed preflight does not leave a partial episode.
sudo -v
mkdir -p "$OUT_DIR"
echo 'episode_id,start_time,end_time,actor,target,technique_id,technique,tactic' > "$GT"
CAPTURE_START="$(iso_now)"
sudo -n tcpdump -i cyberwm0 -n -s 0 -w "$PCAP" 'net 10.77.0.0/24' >/dev/null 2>&1 &
TCPDUMP_PID=$!
sleep 2

BASELINE_ROUNDS="$(random_delay 5 8)"
echo "[baseline] $BASELINE_ROUNDS benign pings: $ACTOR -> $TARGET"
for ((i=1; i<=BASELINE_ROUNDS; i++)); do
  compose_exec "$ACTOR" ping -c 1 "$TARGET_IP" >/dev/null
  sleep 2
done

case "$SCENARIO" in
  benign_ping)
    ;;
  legitimate_ssh)
    sleep "$(random_delay 4 9)"
    run_legitimate_ssh
    ;;
  scan_only)
    sleep "$(random_delay 4 9)"
    run_discovery
    ;;
  failed_guessing)
    sleep "$(random_delay 4 9)"
    run_guessing
    ;;
  one_hop)
    sleep "$(random_delay 4 9)"
    run_discovery
    sleep "$(random_delay 5 10)"
    run_guessing
    sleep "$(random_delay 5 10)"
    run_attack_ssh "$ACTOR" "$PIVOT" "$PIVOT_IP"
    ;;
  two_hop)
    sleep "$(random_delay 4 9)"
    run_discovery
    sleep "$(random_delay 5 10)"
    run_guessing
    sleep "$(random_delay 5 10)"
    run_attack_ssh "$ACTOR" "$PIVOT" "$PIVOT_IP"
    sleep "$(random_delay 5 10)"
    run_attack_ssh "$PIVOT" "$TARGET" "$TARGET_IP"
    ;;
esac

# Retain a quiet tail so the final action is not also the end of state coverage.
sleep 6
CAPTURE_END="$(iso_now)"
cleanup
trap - EXIT

# Metadata is supervision/audit information and must never be fed to the model.
printf 'episode_id,scenario,seed,capture_start,capture_end,actor,pivot,target,window_seconds\n' > "$META"
printf '%s,%s,%s,%s,%s,%s,%s,%s,5\n' \
  "$EPISODE_ID" "$SCENARIO" "$SEED" "$CAPTURE_START" "$CAPTURE_END" \
  "$ACTOR" "$PIVOT" "$TARGET" >> "$META"

# Refuse to announce success when locale/quoting or a future edit corrupts a manifest.
python3 - "$GT" "$META" <<'PY'
import csv
import re
import sys

expected = {sys.argv[1]: 8, sys.argv[2]: 9}
timestamp = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}[+-]\d{2}:\d{2}$")
for path, width in expected.items():
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows or any(len(row) != width for row in rows):
        raise SystemExit(f"invalid CSV width in {path}; expected {width} columns")
    time_columns = (1, 2) if width == 8 else (3, 4)
    for row in rows[1:]:
        for column in time_columns:
            if not timestamp.fullmatch(row[column]):
                raise SystemExit(f"invalid nanosecond timestamp in {path}: {row[column]!r}")
PY

echo "episode written to $OUT_DIR"
echo "  scenario: $SCENARIO (seed=$SEED; roles=$ACTOR->$PIVOT->$TARGET)"
echo "  pcap: $PCAP"
echo "  truth: $GT"
echo "  metadata: $META"
