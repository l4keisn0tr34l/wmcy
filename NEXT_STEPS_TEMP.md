# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Checkpoint

Clean/pushed at `30fc6eb`. Standalone report SHA-256 is `7c62926c14e3053663e90cffc622668dd3877dfc4b1af53bd81da203bddd1b5a`. V4 capture remains interactive.

## Current batch: causal Friday graph sequences

Build train-only public dynamics sequences without inventing a holdout inside one connected capture:

1. Add `scripts/39_build_friday_graph_sequences.py`.
2. Read only canonical observable columns from the 2,102,560-event Friday output.
3. Use exact dense five-second windows, three context states, six future states, and stride five seconds.
4. Select three-host induced-subgraph rosters from context only: most frequent observed context edge plus highest-degree third host; deterministically anonymize slots.
5. Preserve Friday SYN/ACK/RST/FIN counts rather than filling them with zero.
6. Define edge novelty causally from each directed pair's first appearance earlier in the capture, never relative to future/sample-local information.
7. Keep the connected Friday capture as pretraining train only. Do not create a misleading random/temporal validation split.
8. Emit `[N,3,141]`, `[N,6,141]`, and `[N,6,6]` arrays, hashed roster audit, metadata, and finiteness/causality checks.
9. Do not load CIC labels or claim ATT&CK/LM truth.
10. Do not train/tune another model against frozen V3 test. Friday transfer evaluation waits for V4 development/test protocol.
