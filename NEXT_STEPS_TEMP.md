# Active temporary handoff — V6 auth hardening and scenario specification

Updated: 2026-09-12T17:52:00Z
Base commit: b881ee6 (pushed)

## Verified

- Prospective192-episode V6 plan and split are deterministic; 10 contract tests pass.
- Six synthetic auth-parser tests pass.
- Real non-capture password-failure/public-key-success parsing passes.
- Real non-capture dual-zone/flat/inactive SSH+HTTP policy checks pass.
- Both smoke wrappers stop containers in `finally`; no V5/V6 containers or networks remain.
- No V6 PCAP episode, model, scaler, exporter, or test artifact exists; nothing is frozen.

## This code batch

1. Require literal UTC (`Z` or `+00:00`) for auth logs and preserve source nanoseconds in canonical event strings.
2. Verify an actual failed-public-key attempt with an untrusted in-container key, without changing the image.
3. Add a versioned immutable feasibility result rather than overwriting the first smoke result.
4. Specify exact, outcome-independent behavior/timing for all ten V6 scenarios before implementing the capture runtime.
5. Add tests for scenario schedules, same-prefix families, attempted/completed LM semantics, and future margin.

## Hard constraints

- All runtime traffic remains inside the seven-host inventory on10.77.0.0/24.
- Raw OpenSSH text may contain fixed lab usernames/fingerprints; canonical model telemetry must not.
- Controller roles/scenarios/actions/truth remain audit/target data, never observable model input.
- A blocked attempt is not completed LM.
- Do not run PCAP capture, planned episodes, test export, training, or V5 inference.
- No sudo is needed in this batch.
