# UNSW-NB15 Temporal Canonicalization

## Status

Complete for all four local original host-rich CSV files.

```text
2,540,047 valid rows
5 gap-bounded chronological segments
2 connected capture groups
0 invalid timestamps/IPs
0 reversed intervals
```

Artifact root:

```text
outputs/unsw/canonical/
```

Reproduce:

```bash
.venv/bin/python scripts/26_canonicalize_unsw.py \
  /home/paprika/Documents/153/ds/un/CSV\ Files/UNSW-NB15_{1,2,3,4}.csv \
  --features /home/paprika/Documents/153/ds/un/CSV\ Files/NUSW-NB15_features.csv \
  --out-dir outputs/unsw/canonical
```

## Input

The original files are headerless. The adapter uses the accompanying 49-field definition with Latin-1 decoding. Relevant observable fields include source/destination IP and port, protocol, service, connection state, Unix-second `Stime/Ltime`, packets, bytes, and duration.

Raw ordering is not chronological: each file has approximately 52,000–108,000 timestamp inversions. Stable sorting is mandatory before temporal modeling.

## Transformation and output

Each source file is sorted by start time, end time, and original row number. Gaps over one hour create separate segments so dense graph construction never fills multi-week gaps with invented quiet states.

Every segment contains:

```text
observations.csv.gz       defender-observable canonical fields
row_ground_truth.csv.gz   raw attack category/binary truth and provenance
```

The adapter does not fabricate unavailable TCP flag counts, ATT&CK techniques, or lateral-movement truth.

## Capture-group leakage control

Source-file boundaries overlap by tens of seconds. Connected intervals are assigned to the same capture group:

```text
unsw_capture_00: unsw_01_00 + unsw_02_00
unsw_capture_01: unsw_02_01 + unsw_03_00 + unsw_04_00
```

These components must never be split independently across train/test. With only two public capture groups, UNSW is currently best used for telemetry pretraining and a coarse domain holdout—not many random folds.

## Intended use

Use UNSW for:

- chronological graph-dynamics pretraining;
- broader benign/attack-regime traffic variation;
- domain-shift diagnostics.

Do not use its broad attack category as exact LM or ATT&CK progression truth. Exact LM/ATT&CK semantics remain from controlled lab actions.
