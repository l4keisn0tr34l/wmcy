#!/usr/bin/env bash
set -euo pipefail

# Controlled ATT&CK-labelled episode for the private Docker lab.
# The script writes ground truth because it knows the exact behavior it executes.
# It does NOT derive labels from network features.

EPISODE_ID="${1:-lab_001}"
OUT_DIR="${2:-./episodes/$EPISODE_ID}"
mkdir -p "$OUT_DIR"
GT="$OUT_DIR/ground_truth.csv"
PCAP="$OUT_DIR/network.pcap"

echo 'episode_id,start_time,end_time,actor,target,technique_id,technique,tactic' > "$GT"

iso_now() { date --iso-8601=ns; }
record_event() {
  local start="$1" end="$2" actor="$3" target="$4" tid="$5" tech="$6" tactic="$7"
  printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$EPISODE_ID" "$start" "$end" "$actor" "$target" "$tid" "$tech" "$tactic" >> "$GT"
}

cleanup() {
  if [[ -n "${TCPDUMP_PID:-}" ]]; then
    sudo kill "$TCPDUMP_PID" 2>/dev/null || true
    wait "$TCPDUMP_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# Capture the bridge directly so timestamps retain sub-second precision.
sudo tcpdump -i cyberwm0 -n -s 0 -w "$PCAP" 'net 10.77.0.0/24' >/dev/null 2>&1 &
TCPDUMP_PID=$!
sleep 2

echo '[1/5] benign baseline (30s)'
docker compose exec -T ws1 bash -lc 'for i in {1..10}; do ping -c 1 10.77.0.40 >/dev/null; sleep 3; done'

echo '[2/5] T1046 Network Service Discovery'
START=$(iso_now)
docker compose exec -T ws1 nmap -sT -Pn -p 22,80,443,445,3389 10.77.0.0/24 >/dev/null
END=$(iso_now)
record_event "$START" "$END" ws1 internal_subnet T1046 'Network Service Discovery' Discovery
sleep 10

echo '[3/5] T1110.001 Password Guessing against srv1 SSH'
START=$(iso_now)
for pw in wrong1 wrong2 wrong3; do
  docker compose exec -T ws1 bash -lc \
    "sshpass -p '$pw' ssh -o StrictHostKeyChecking=no -o PubkeyAuthentication=no -o PreferredAuthentications=password -o ConnectTimeout=2 lab@10.77.0.30 'true'" \
    >/dev/null 2>&1 || true
done
END=$(iso_now)
record_event "$START" "$END" ws1 srv1 T1110.001 'Password Guessing' 'Credential Access'
sleep 10

echo '[4/5] T1021.004 SSH lateral movement ws1 -> srv1'
START=$(iso_now)
docker compose exec -T ws1 bash -lc \
  "sshpass -p labpass ssh -o StrictHostKeyChecking=no lab@10.77.0.30 'hostname && id'" >/dev/null
END=$(iso_now)
record_event "$START" "$END" ws1 srv1 T1021.004 SSH 'Lateral Movement'
sleep 10

echo '[5/5] T1021.004 second hop srv1 -> srv2'
START=$(iso_now)
docker compose exec -T srv1 bash -lc \
  "sshpass -p labpass ssh -o StrictHostKeyChecking=no lab@10.77.0.40 'hostname && id'" >/dev/null
END=$(iso_now)
record_event "$START" "$END" srv1 srv2 T1021.004 SSH 'Lateral Movement'
sleep 5

cleanup
trap - EXIT

echo "episode written to $OUT_DIR"
echo "  pcap: $PCAP"
echo "  truth: $GT"
