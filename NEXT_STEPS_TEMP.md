# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## V4 capture verified

All 36 planned `lab_073`–`lab_108` raw and derived episode files exist. General episode validation plus V4 action semantics pass for 36/36. Containers have no residual firewall rules. Corpus totals:

```text
episodes:          36
five-second states:828 (23 each)
observations:      2,559
ATT&CK events:     90
chosen actions:    24
zero-traffic states:399
capture duration:  ~120.007–120.014 s
```

The pandas `metadata.pivot` validator bug was fixed to bracket-based column access; it was not a data error.

## Current batch: V4 action-train sequences only

1. Add `scripts/42_build_v4_action_sequences.py`.
2. Default to `--split train`; load only the 12 V4 action-training episode directories.
3. Require explicit `--unlock-test` before any V4 action-test episode can be read.
4. For each action, use the three complete five-second states ending immediately before the intervention boundary.
5. Put the action/result window and next five windows in the six-state target.
6. Emit observable context plus separately known action type `[permit,block]` and action directed-pair one-hot.
7. Emit future graph/edge/LM/ATT&CK/pair targets separately.
8. Assert no completed LM in context, action time is about 0.2 s after the future grid boundary, action precedes SSH outcome, permit completes LM, and block records attempt but not completion.
9. Preserve graph feature contract and host/pair equivariance; action-pair augmentation must later follow host relabeling.
10. Write train arrays/audit atomically and inspect shapes/finiteness/pair balance.
11. Do not load, build, or predict on the 12 action-test or 12 passive/direct test episodes yet.
