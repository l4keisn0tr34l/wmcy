# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current verified checkpoint

Graph RSSM and semantic follow-ups are committed at `6703e8b`:

```text
Graph dynamics: state MAE 0.252, active MAE 0.895, edge AP 0.394
Graph pair:    0.357 (5/14), exactly stable across all host relabelings
Rich LM head:  F1/AP 0.667/0.744, pre-first F1 0.667
```

Updated report: `/home/paprika/Downloads/rssm_eod_report.html`, SHA-256 `d6ba73be7f00a71ff72a53de295d731c3b32e5b49d0e9bef12f6a759aa2c3216`.

## Verified local public-data facts

UNSW-NB15 original files contain 2,540,047 rows total, source/destination IP and port, Unix-second `Stime/Ltime`, packets/bytes/duration, services/states, and separate attack category/label fields. Rows are not chronological (52k–108k inversions per file), so sorting/segmenting is mandatory. Internal range is documented/observed as `149.171.126.0/24`. Attack labels do not provide clean lateral-movement progression.

CICIDS2018 has ten local processed files with second timestamps, but only `Thuesday-20-02-2018...csv` retains Src/Dst IP in this copy. CICIDS2017 ML files omit IPs; TrafficLabelling copies retain them but have minute timestamps. Friday PCAP remains precise telemetry.

## Current code-writing batch: UNSW canonical temporal adapter

Implement `scripts/26_canonicalize_unsw.py`:

1. Read headerless original UNSW files using the official local 49-field definition and Latin-1 encoding.
2. Preserve raw files unchanged.
3. Map observable primitives to the existing canonical event contract:
   - Stime UTC timestamp, src/dst IP and ports, protocol/service/state;
   - source/destination packets and bytes;
   - duration.
4. Keep `attack_cat` and `Label` only in a separate row-ground-truth file.
5. Do not fabricate TCP flags unavailable in UNSW.
6. Stable-sort by Unix start/end time because raw rows are heavily out of order.
7. Split each source file at configurable large timestamp gaps so dense state generation cannot fill multi-day gaps with artificial empty states.
8. Reset event IDs inside each generated segment and preserve source-row provenance only in ground truth/audit metadata, not model input.
9. Write per-segment canonical observations, separate truth, and a profile/manifest JSON.
10. Smoke-test on a bounded row count, then process all four local files.

## Validation

- observations contain no label/attack category;
- truth and observations have one-to-one event IDs;
- timestamps monotonic within segments and retain one-second precision without fabrication;
- no segment crosses configured large gaps;
- IP identities and observable quantities are present;
- profile records unavailable flag fields;
- inspect generated output, compile, `git diff --check`.

## Following batch

Build three-host chronological induced-subgraph pretraining sequences from UNSW segments, anonymize host slots consistently per sequence/episode, pretrain graph dynamics without attack labels, then fine-tune on lab truth. In parallel add matched scan/guessing hard-negative lab episodes for semantics.
