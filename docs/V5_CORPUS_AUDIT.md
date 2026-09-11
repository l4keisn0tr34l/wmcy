# V5 corpus consistency audit

## Status

**PASS: all 80 planned episodes are raw-complete, derived-complete, and independently pass the general episode validator and strict V5 contract validator.** This was dataset QA across all splits, not model inference, preprocessing fitting, model selection, or test evaluation.

Reproducible artifacts:

```text
scripts/50_audit_v5_corpus.py
outputs/mvp_v5/corpus_audit/audit.json
outputs/mvp_v5/corpus_audit/episode_summary.csv
outputs/mvp_v5/corpus_audit/paired_family_summary.csv
outputs/mvp_v5/corpus_audit/background_failures.csv
```

Run once on the immutable corpus:

```bash
.venv/bin/python scripts/50_audit_v5_corpus.py --deep-validate
```

## Duration and censoring result

The historical unequal-duration shortcut is absent.

```text
episodes:                              80 / 80
raw capture duration minimum:          150.000094 s
raw capture duration maximum:          150.000615 s
raw capture duration mean:             150.000163 s
attack-vs-benign mean absolute gap:    0.0000154 s
maximum within-family duration gap:    0.000456 s
packet drops:                          0
```

Every episode has exactly 29 dense, complete five-second states. The 145 seconds of model-facing state coverage is expected: capture begins at an arbitrary sub-five-second phase, and the state builder excludes both partial boundary bins. It does not pad or label partial time as complete. Because every 150-second capture loses exactly five seconds across the two partial boundaries, every accepted episode retains the same 29 complete states.

Thus neither episode duration nor state count reveals benign/attack outcome. PCAP byte size, packet count, observation count, and zero-state count vary because they are telemetry consequences of profile/scenario activity, not censoring.

## Paired-family result

All 40 paired families contain exactly two episodes with:

- one common seed;
- one common split;
- one common background profile;
- 29 states in both alternatives;
- no raw-duration discrepancy above 0.456 ms.

Pairs are intentionally independent captures, not duplicated packets. Family-seeded schedules match nominal elapsed decision times, but each capture has an independent absolute start phase. Aligning the forecast to the next absolute five-second boundary therefore produces residual paired timing differences:

```text
scan-action maximum cutoff-offset gap:       3.3595 s
credential-action maximum cutoff-offset gap: 2.7230 s
maximum paired forecast-state-index gap:     1 state
families with one-index action mismatch:     2 / 20 action families
```

This is not a horizon or length mismatch: every action sample uses the three immediately preceding complete states and six immediately following complete states. It does mean paired contexts are similar rather than identical, and future analysis must not present them as exact randomized counterfactual twins.

## Timing and semantic result

Observed event offsets remain within the frozen nuisance schedule:

```text
T1046 start:       20–28 s
T1110.001 start:   43–49 s
T1021.004 start:   82.034–107.906 s
action decisions: 80.000–103.000 s
```

All 40 chosen actions satisfy:

```text
decision <= forecast cutoff < effective action <= recorded action end
minimum capture margin after cutoff: 42.294828 s
required future horizon:              30 s
```

Permit episodes contain completed `T1021.004` / Lateral Movement truth. Block episodes retain attempted `T1021.004` truth but zero completed LM. Benign SSH has no malicious ground truth. ATT&CK remains a separate target layer and is not present in observable state vectors.

## Background result

Every episode has background records from all five hosts and the expected quiet/web/admin/mixed profile kinds. Background begins within 5.293 seconds and its last event begins no more than 15.840 seconds before capture end. The maximum gap is compatible with the bounded 8–14-second quiet/admin schedule plus command runtime; there is no profile-wide artificial quiet tail.

Thirteen nonzero background command statuses were inspected:

- 12 are post-effective `ssh` failures on the exact source/target pair intentionally blocked by the defender action;
- one is a bounded timeout for a command launched at the capture tail.

These are expected action/deadline consequences, not uncontrolled episode failures. The block-related failures also mean the action changes legitimate administrative traffic on the selected pair, not only the focal attacker command. That is a real consequence of the source-specific firewall action and should be disclosed when interpreting state differences.

## Quarantines

Excluded raw directories remain outside the plan:

```text
_quarantine_v5_smoke_001_reboot_20260911T234648
_quarantine_lab_158_ssh_transport_20260912T015503
```

The first was manually interrupted. The second rejected an SSH transport exit (`255`) rather than falsely recording it as password rejection (`5`). `lab_158` was subsequently recaptured from time zero and validates. Neither quarantine was processed or used.

## What this audit establishes

Observed facts:

- equal raw duration and equal complete-state count across accepted episodes;
- whole paired families remain within their assigned splits;
- complete action horizons and correct attempt/completion semantics;
- zero packet drops and passing structural/provenance checks;
- no recurrence of the historical shorter-attack-episode censoring shortcut.

It does **not** establish:

- packet-identical paired prefixes;
- unseen-topology generalization (V5 is one flat five-host topology);
- enterprise realism;
- calibrated uncertainty;
- beneficial Friday/V4 transfer;
- model performance of any kind.

Those require a separate train/validation-only protocol and one later sealed evaluation.
