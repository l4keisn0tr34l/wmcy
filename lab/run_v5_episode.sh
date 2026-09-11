#!/usr/bin/env bash
set -euo pipefail

# Five-host V5 traffic generation. Every adversarial command is restricted to
# the isolated Docker subnet 10.77.0.0/24.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE=(docker compose -p cyberwm_v5 -f "$SCRIPT_DIR/docker-compose-v5.yml")
HOSTS=(ws1 ws2 srv1 srv2 admin1)

usage() {
  cat <<'EOF'
Usage: ./run_v5_episode.sh EPISODE_ID [options]

Options:
  --scenario NAME       background_only | legitimate_ssh | matched_legitimate_ssh |
                        credential_one_hop | scan_guess_then_stop | one_hop |
                        scan_guess_action_permit | scan_guess_action_block |
                        credential_action_permit | credential_action_block
  --background NAME     quiet | web | admin | mixed
  --seed INTEGER
  --duration-seconds N  fixed at 150 for V5
  --out-dir PATH
EOF
}

[[ $# -ge 1 ]] || { usage >&2; exit 2; }
EPISODE_ID="$1"; shift
SCENARIO="background_only"; BACKGROUND_PROFILE="quiet"; SEED="$(date +%s)"
DURATION_SECONDS=150; OUT_DIR="$SCRIPT_DIR/episodes/$EPISODE_ID"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO="${2:?missing scenario}"; shift 2 ;;
    --background) BACKGROUND_PROFILE="${2:?missing background}"; shift 2 ;;
    --seed) SEED="${2:?missing seed}"; shift 2 ;;
    --duration-seconds) DURATION_SECONDS="${2:?missing duration}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:?missing output}"; shift 2 ;;
    --help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
case "$SCENARIO" in
  background_only|legitimate_ssh|matched_legitimate_ssh|credential_one_hop|scan_guess_then_stop|one_hop|scan_guess_action_permit|scan_guess_action_block|credential_action_permit|credential_action_block) ;;
  *) echo "invalid V5 scenario: $SCENARIO" >&2; exit 2 ;;
esac
case "$BACKGROUND_PROFILE" in quiet|web|admin|mixed) ;; *) echo "invalid background profile" >&2; exit 2 ;; esac
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "seed must be non-negative integer" >&2; exit 2; }
[[ "$DURATION_SECONDS" == 150 ]] || { echo "V5 duration is frozen at 150 seconds" >&2; exit 2; }
[[ "$EPISODE_ID" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid episode ID" >&2; exit 2; }

GT="$OUT_DIR/ground_truth.csv"; PCAP="$OUT_DIR/network.pcap"
META="$OUT_DIR/episode_metadata.csv"; ACTIONS="$OUT_DIR/defender_actions.csv"
if [[ -e "$GT" || -e "$PCAP" || -e "$META" || -e "$ACTIONS" ]]; then
  echo "refusing to overwrite raw episode files in $OUT_DIR" >&2; exit 1
fi

iso_now() { date '+%Y-%m-%dT%H:%M:%S.%N%:z'; }
record_event() {
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' "$EPISODE_ID" "$1" "$2" "$3" "$4" "$5" "$6" "$7" >> "$GT"
}
record_action() {
  printf '%s,%s,%s,%s,%s,%s,true,%s\n' "$EPISODE_ID" "$1" "$2" "$3" "$4" "$5" "$6" >> "$ACTIONS"
}
host_ip() {
  case "$1" in
    ws1) echo 10.77.0.20 ;; ws2) echo 10.77.0.25 ;; srv1) echo 10.77.0.30 ;;
    srv2) echo 10.77.0.40 ;; admin1) echo 10.77.0.50 ;;
    *) echo "unknown V5 host: $1" >&2; return 1 ;;
  esac
}
compose_exec() { local host="$1"; shift; "${COMPOSE[@]}" exec -T "$host" "$@"; }

mapfile -t ROLE_HOSTS < <(python3 - "$SEED" <<'PY'
import random, sys
hosts = ["ws1", "ws2", "srv1", "srv2", "admin1"]
random.Random(int(sys.argv[1])).shuffle(hosts)
print(*hosts, sep="\n")
PY
)
ACTOR="${ROLE_HOSTS[0]}"; PIVOT="${ROLE_HOSTS[1]}"; TARGET="${ROLE_HOSTS[2]}"
BG1="${ROLE_HOSTS[3]}"; BG2="${ROLE_HOSTS[4]}"
ACTOR_IP="$(host_ip "$ACTOR")"; PIVOT_IP="$(host_ip "$PIVOT")"; TARGET_IP="$(host_ip "$TARGET")"
BG1_IP="$(host_ip "$BG1")"; BG2_IP="$(host_ip "$BG2")"
RANDOM=$((SEED % 32768))
random_attempts() { echo $((3 + RANDOM % 3)); }

align_to_next_state_boundary() {
  sleep "$(python3 - <<'PY'
import time
print(f"{5.0 - (time.time() % 5.0) + 0.2:.6f}")
PY
)"
}
sleep_until_elapsed() {
  python3 - "$CAPTURE_START_NS" "$1" <<'PY'
import sys, time
remaining = float(sys.argv[2]) - (time.time_ns() - int(sys.argv[1])) / 1e9
if remaining > 0: time.sleep(remaining)
PY
}

run_discovery() {
  local start end; echo "[action] discovery $ACTOR -> isolated subnet"
  start="$(iso_now)"
  compose_exec "$ACTOR" nmap -sT -Pn -p 22,80,443,445,3389,8080 10.77.0.0/24 >/dev/null
  end="$(iso_now)"; record_event "$start" "$end" "$ACTOR" internal_subnet T1046 'Network Service Discovery' Discovery
}
run_guessing() {
  local attempts start end; attempts="$(random_attempts)"
  echo "[action] $attempts failed passwords $ACTOR -> $PIVOT"
  start="$(iso_now)"
  for ((i=1; i<=attempts; i++)); do
    compose_exec "$ACTOR" bash -lc "sshpass -p wrong$i ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=2 lab@$PIVOT_IP true" >/dev/null 2>&1 || true
  done
  end="$(iso_now)"; record_event "$start" "$end" "$ACTOR" "$PIVOT" T1110.001 'Password Guessing' 'Credential Access'
}
run_attack_ssh() {
  local start end; echo "[action] completed SSH $ACTOR -> $PIVOT"
  start="$(iso_now)"
  compose_exec "$ACTOR" bash -lc "sshpass -p labpass ssh -o StrictHostKeyChecking=no lab@$PIVOT_IP 'hostname && id'" >/dev/null
  end="$(iso_now)"; record_event "$start" "$end" "$ACTOR" "$PIVOT" T1021.004 SSH 'Lateral Movement'
}
run_legitimate_ssh() {
  echo "[action] legitimate SSH $ACTOR -> $PIVOT"
  compose_exec "$ACTOR" bash -lc "sshpass -p labpass ssh -o StrictHostKeyChecking=no lab@$PIVOT_IP 'hostname && id'" >/dev/null
}
run_blocked_ssh() {
  local start end; echo "[action] blocked SSH attempt $ACTOR -> $PIVOT"
  start="$(iso_now)"
  if compose_exec "$ACTOR" bash -lc "sshpass -p labpass ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 lab@$PIVOT_IP 'hostname && id'" >/dev/null 2>&1; then
    echo "blocked SSH unexpectedly succeeded" >&2; return 1
  fi
  end="$(iso_now)"; record_event "$start" "$end" "$ACTOR" "$PIVOT" T1021.004 SSH 'Lateral Movement Attempt'
}
record_permit() { local now; now="$(iso_now)"; record_action "$now" "$now" permit_ssh "$ACTOR" "$PIVOT" tcp_22_source_permitted; }
apply_block() {
  local start end; start="$(iso_now)"
  compose_exec "$PIVOT" iptables -I INPUT 1 -p tcp -s "$ACTOR_IP" --dport 22 -j REJECT --reject-with tcp-reset
  BLOCK_INSTALLED=1; end="$(iso_now)"
  record_action "$start" "$end" block_ssh "$ACTOR" "$PIVOT" tcp_22_source_rejected
}

BACKGROUND_PIDS=()
spawn_quiet() {
  compose_exec "$BG1" bash -lc "for i in \$(seq 1 12); do ping -c 1 -W 1 $BG2_IP >/dev/null || true; sleep 8; done" >/dev/null 2>&1 & BACKGROUND_PIDS+=("$!")
}
spawn_web() {
  compose_exec "$BG1" bash -lc "for i in \$(seq 1 30); do curl -fsS --max-time 2 http://$BG2_IP:8080/ >/dev/null || true; sleep 3; done" >/dev/null 2>&1 & BACKGROUND_PIDS+=("$!")
}
spawn_admin() {
  compose_exec "$BG1" bash -lc "for i in \$(seq 1 6); do sshpass -p labpass ssh -o StrictHostKeyChecking=no -o ConnectTimeout=3 lab@$BG2_IP 'true' >/dev/null 2>&1 || true; sleep 15; done" >/dev/null 2>&1 & BACKGROUND_PIDS+=("$!")
}
start_background() {
  echo "[background] profile=$BACKGROUND_PROFILE hosts=$BG1->$BG2"
  case "$BACKGROUND_PROFILE" in
    quiet) spawn_quiet ;; web) spawn_web ;; admin) spawn_admin ;;
    mixed) spawn_quiet; spawn_web; spawn_admin ;;
  esac
}

BLOCK_INSTALLED=0
cleanup() {
  if [[ "$BLOCK_INSTALLED" -eq 1 ]]; then
    compose_exec "$PIVOT" iptables -D INPUT -p tcp -s "$ACTOR_IP" --dport 22 -j REJECT --reject-with tcp-reset >/dev/null 2>&1 || true
    BLOCK_INSTALLED=0
  fi
  for pid in "${BACKGROUND_PIDS[@]:-}"; do kill "$pid" >/dev/null 2>&1 || true; done
  if [[ -n "${TCPDUMP_PID:-}" ]]; then sudo -n kill "$TCPDUMP_PID" 2>/dev/null || true; wait "$TCPDUMP_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

running="$("${COMPOSE[@]}" ps --services --status running | wc -l)"
[[ "$running" -eq 5 ]] || { echo "all five V5 services must be running" >&2; exit 1; }
[[ -d /sys/class/net/cyberwmv5 ]] || { echo "missing isolated cyberwmv5 bridge" >&2; exit 1; }
sudo -v
mkdir -p "$OUT_DIR"
echo 'episode_id,start_time,end_time,actor,target,technique_id,technique,tactic' > "$GT"
echo 'episode_id,start_time,end_time,action,source,target,known_at_forecast_time,details' > "$ACTIONS"
CAPTURE_START="$(iso_now)"; CAPTURE_START_NS="$(date +%s%N)"
CAPTURE_DEADLINE_NS=$((CAPTURE_START_NS + DURATION_SECONDS * 1000000000))
sudo -n tcpdump -i cyberwmv5 -n -s 0 -w "$PCAP" 'net 10.77.0.0/24' >/dev/null 2>&1 & TCPDUMP_PID=$!
sleep 1; align_to_next_state_boundary; start_background

case "$SCENARIO" in
  scan_guess_then_stop|one_hop|scan_guess_action_permit|scan_guess_action_block)
    sleep_until_elapsed 25; run_discovery; sleep_until_elapsed 50; run_guessing ;;
esac
sleep_until_elapsed 85
case "$SCENARIO" in
  background_only|scan_guess_then_stop) ;;
  legitimate_ssh|matched_legitimate_ssh) run_legitimate_ssh ;;
  credential_one_hop|one_hop) run_attack_ssh ;;
  scan_guess_action_permit|credential_action_permit)
    align_to_next_state_boundary; record_permit; run_attack_ssh ;;
  scan_guess_action_block|credential_action_block)
    align_to_next_state_boundary; apply_block; run_blocked_ssh ;;
esac

remaining="$(python3 - "$CAPTURE_DEADLINE_NS" <<'PY'
import sys, time
value = (int(sys.argv[1]) - time.time_ns()) / 1e9
if value < 35: raise SystemExit(f"less than 35 seconds remain after outcome: {value:.3f}")
print(f"{value:.9f}")
PY
)"
echo "[capture] retaining background until fixed 150s deadline"
sleep "$remaining"; CAPTURE_END="$(iso_now)"; cleanup; trap - EXIT

DEFENDER_ACTION=none
case "$SCENARIO" in *_action_permit) DEFENDER_ACTION=permit_ssh ;; *_action_block) DEFENDER_ACTION=block_ssh ;; esac
printf 'episode_id,scenario,seed,capture_start,capture_end,actor,pivot,target,background_host_1,background_host_2,background_profile,topology_profile,node_count,window_seconds,planned_capture_duration_seconds,defender_action\n' > "$META"
printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,flat_five_host,5,5,%s,%s\n' \
  "$EPISODE_ID" "$SCENARIO" "$SEED" "$CAPTURE_START" "$CAPTURE_END" "$ACTOR" "$PIVOT" "$TARGET" "$BG1" "$BG2" "$BACKGROUND_PROFILE" "$DURATION_SECONDS" "$DEFENDER_ACTION" >> "$META"

python3 - "$GT" "$META" "$ACTIONS" "$SCENARIO" <<'PY'
import csv, re, sys
expected = {sys.argv[1]: 8, sys.argv[2]: 16, sys.argv[3]: 8}
timestamp = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{9}[+-]\d{2}:\d{2}$")
for path, width in expected.items():
    with open(path, newline="", encoding="utf-8") as handle: rows = list(csv.reader(handle))
    if not rows or any(len(row) != width for row in rows): raise SystemExit(f"invalid CSV width: {path}")
    columns = (1, 2) if width == 8 else (3, 4)
    for row in rows[1:]:
        for column in columns:
            if not timestamp.fullmatch(row[column]): raise SystemExit(f"invalid timestamp in {path}")
with open(sys.argv[3], newline="", encoding="utf-8") as handle: actions = list(csv.DictReader(handle))
expected_action = "permit_ssh" if sys.argv[4].endswith("_action_permit") else "block_ssh" if sys.argv[4].endswith("_action_block") else None
if expected_action is None and actions: raise SystemExit("unexpected action row")
if expected_action and (len(actions) != 1 or actions[0]["action"] != expected_action): raise SystemExit("missing/wrong action")
PY

echo "V5 episode written: $OUT_DIR"
echo "  scenario=$SCENARIO background=$BACKGROUND_PROFILE seed=$SEED roles=$ACTOR->$PIVOT->$TARGET bg=$BG1->$BG2"
