# Active temporary handoff — V6 real policy feasibility

Updated: 2026-09-12T17:41:00Z

## Current verified state

- Prospective V6 plan: 192 episodes / 96 families; 10 contract tests pass.
- OpenSSH parser: 6 synthetic tests pass.
- Isolated seven-container V6 image/compose builds and stops cleanly.
- A durable non-capture smoke verified one password rejection and one key success from real Docker/OpenSSH UTC logs; canonical events omit username and key fingerprint.
- All seven smoke containers used one image ID. No V5 source/image/checkpoint or sealed output was modified.
- No V6 episode, PCAP, scaler, model, or test artifact exists. Nothing is frozen.

## This code batch

1. Implement a reproducible non-capture policy-realization smoke.
2. Apply generated inventory-only SSH/HTTP rejection rules inside the isolated V6 network.
3. Verify direct dual-zone service access is blocked while each permitted jump-role leg succeeds.
4. Verify the five-active flat profile permits active-to-active access and blocks inactive slots.
5. Tear down in `finally`, retain only sanitized structured results, and fail closed on any mismatch.
6. Add synthetic/background allowlist checks if required by the smoke findings.

## Hard constraints

- Targets must remain inside 10.77.0.0/24 and the seven-host inventory.
- This is topology feasibility, not a capture episode and not model evidence.
- Do not use planned test episode data, run PCAP capture, or inspect any sealed V5 test pathway.
- Do not modify frozen V5 files or images.
- Do not request sudo or begin corpus capture in this batch.
