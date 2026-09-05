# Controlled lateral-movement data generator

This private Docker network gives us exact telemetry + exact ATT&CK ground truth.

## Why we need it

CICIDS2017 is useful for generic network dynamics but does not provide clean, precise, successful lateral-movement trajectories. This lab creates the missing data without inventing feature vectors.

## Run

```bash
cd lab
docker compose up -d --build
./run_episode.sh lab_001
```

The episode deliberately performs, only inside `10.77.0.0/24`:

1. benign traffic;
2. `T1046` Network Service Discovery;
3. `T1110.001` Password Guessing;
4. `T1021.004` SSH lateral movement from `ws1` to `srv1`;
5. a second `T1021.004` hop from `srv1` to `srv2`.

It produces:

```text
episodes/lab_001/network.pcap
episodes/lab_001/ground_truth.csv
```

Then convert the packet capture to exact-time canonical events:

```bash
python ../scripts/06_pcap_to_canonical.py \
  episodes/lab_001/network.pcap \
  --bucket-seconds 1 \
  --dataset-id lab_001 \
  --out episodes/lab_001/observations.csv.gz
```

Build five-second graph states:

```bash
python ../scripts/03_build_graph_states.py \
  episodes/lab_001/observations.csv.gz \
  --window 5s \
  --internal-cidr 10.77.0.0/24 \
  --out-dir episodes/lab_001/states
```

Align ATT&CK truth to those state windows:

```bash
python ../scripts/05_align_ground_truth.py \
  episodes/lab_001/states/global_states.csv \
  episodes/lab_001/ground_truth.csv \
  --window-seconds 5 \
  --out episodes/lab_001/state_ground_truth.csv
```

At this point we have a real world-model episode: chronological graph states + future states + exact ATT&CK/lateral-movement truth.
