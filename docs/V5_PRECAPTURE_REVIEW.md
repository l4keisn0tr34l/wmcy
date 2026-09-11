# V5 pre-capture review and five-host contract

## Status

Implementation/methodology review and two real smokes completed before capture. **The subsequently frozen 80-episode corpus is now complete and passes the post-capture consistency audit.** This document preserves the pre-capture rationale; see `docs/V5_CORPUS_AUDIT.md` for observed corpus evidence.

No V3/V4 test predictions or tuning were performed. V5 model/evaluation protocols are not frozen. The capture-only freeze exists at `configs/mvp_v5_capture_freeze.json` (SHA-256 `53069f98ed2f8e686f973da6aa6f62f22f4d9b735def1da9868e7247783a20df`). V5 model/evaluation protocols remain explicitly unfrozen.

## Corrections and why they matter

| Stage | Input → transformation → output | Failure prevented / excluded information | Remaining limitation |
|---|---|---|---|
| Capture scheduling | Family seed → independent nuisance schedule: discovery 20–28s, guessing 43–50s, decision 80–103s | Removes one fixed intervention time; schedules never enter model inputs | Still a narrow scheduled lab distribution |
| Capture order | Opaque episode ID → fixed hash order, recorded at freeze | Avoids split-ordered and consistently permit-before-block acquisition | Does not make independent runs packet-identical |
| Background | All five known hosts + family profile/seed → bounded ping/HTTP/admin activity until capture end−2s | Avoids background-only host identities revealing actor roles, and the old ~100s quiet tail | Traffic remains synthetic; two families/cohort cannot span all four profiles |
| Profile assignment | Same 80 families/splits → rotated validation/test profiles | Removes the repeated quiet/web versus admin/mixed cohort partition | Residual within-split cohort/profile association remains; report stratified results |
| Isolation/cleanup | Owned Compose project → fresh containers/network each episode; internal bridge and exact inventory check; HTTP serves one fixed file | No internet-facing scans, no host multicast/gateway telemetry, no inherited SSH/background/firewall state | Docker namespace isolation is not enterprise realism |
| Action boundary | Controller chooses action before next 5s boundary; application begins ~0.2s after it | Action type/pair exists by forecast cutoff; no action-effect packets in context | Chosen action is exogenous; not inferred intent or policy optimization |
| Observation export | Strict global/node/edge schemas → fixed inventory tensors and masks | No raw IP numeric features, future roster selection, truth columns, unknown silently discarded hosts, missing/partial time bins | Global and local traffic describe only inventory-to-inventory IPv4 |
| Target export | Exact separate truth → half-open time alignment and point-event handling | Blocked `T1021.004` remains an attempt, not completed LM; labels never enter context | Scenario truth is controlled intent, not independently inferred from traffic |
| Freeze | Passing real smokes + source/image/raw hashes → exclusive capture-freeze JSON | Missing smokes, changed reviewed files/images, or changed smoke artifacts block corpus startup | This is reproducibility gating, not a cryptographic authorization boundary |

Capture bounds are an **interior measured interval** from tcpdump readiness to stop request, not process-launch/exit latency. Commands have timeouts; prefix overrun, unexpected SSH return status, capture drops, inadequate future margin, and failed cleanup reject the episode. Metadata is installed only after firewall cleanup, worker completion, and container stop. Interrupted raw directories are preserved for quarantine; never resumed mid-episode.

All five hosts generate benign traffic independently of assigned malicious roles. The legacy `target`/two `background_host` metadata slots remain a seed-order audit, not claims that only two hosts generate background or that a second malicious hop occurs. V5 still executes **one actor→pivot SSH hop**.

## Concrete sequence contract

`scripts/47_build_v5_sequences.py` uses `src/cyberwm/v5_sequences.py` and an explicit feature allowlist in `src/cyberwm/v5_contract.py`:

- 15 global + 5×18 node + 20×12 directed-edge features = **345**.
- `context_states`: `[N,3,345]`, 15 seconds.
- `future_states`: `[N,6,345]`, 30 seconds.
- Future edge presence / completed-LM pairs: `[N,6,20]`.
- Technique targets: `[N,6,3]`; completed LM: `[N,6]` and horizon-level `[N]`.
- Action-only arrays: `action_type [N,2]`, `action_pair [N,20]`.
- Slot IPs and scenario/family/profile/timing provenance live in metadata/audit files, not feature vectors.

Modes:

1. `passive`: sliding windows from **non-intervention episodes only**. Test-only intent-probe episodes cannot enter training because whole-family split selection is explicit.
2. `action`: one complete action-aligned window per intervention, with separate chosen type/pair.
3. `passive_action`: identical aligned telemetry/targets, without either action input. This supports a future fair observability comparison; no model evaluation is performed by export.

Default split is train. Validation is explicit. Test requires `--split test --unlock-test` only after a separate model/protocol freeze. Train/validation/smoke exporters never inspect test episode contents. Outputs are raw features: **no scaler is fitted here**. Fit one shared-slot V5 scaler using train contexts only in the future trainer; never reuse the old 141-dimensional scaler. Atomic directory installation refuses existing outputs.

Example after actual data exist:

```bash
.venv/bin/python scripts/47_build_v5_sequences.py --split train --mode passive
.venv/bin/python scripts/47_build_v5_sequences.py --split train --mode action
.venv/bin/python scripts/47_build_v5_sequences.py --split validation --mode action
```

Output directories: `outputs/mvp_v5/sequences/<mode>/<split>/`, each with `<split>.npz`, `sample_manifest.csv`, and `feature_metadata.json`.

## Verified, not hypothesized

- Twelve synthetic regressions pass, including shutdown/sleep inhibitor enforcement: shapes, silent inventory, benign windows, completed point-event LM, blocked attempt semantics, truth/future input independence, late action/short horizon rejection, unknown-host/extra-label/reference rejection, test lock, immutable export, freeze gate, and permutation/schedule checks.
- On local CUDA, all 120 host/action-pair permutations have maximum float32 discrepancy `2.98e-8` initially and `3.35e-8` in the final repeat.
- Exact CPU future-perturbation discrepancy is zero. CUDA discrepancy `2.98e-8` is within the `1e-6` test tolerance; repeated identical CUDA inputs also differed at that scale because of floating-point reductions. Do not report CUDA bitwise equality.
- Gradients tested are finite. Shared-slot scaler normalization commutes with host relabeling.
- Actual frozen Friday and V4-scratch parameter dictionaries load into five-host models: Friday has only 11 expected missing action-layer parameter tensors; V4 has none. Old scalers were not reused. **Shape compatibility is not empirical transfer evidence.** No frozen model was evaluated on old data.
- Shell syntax, Python compilation, Compose configuration, BPF compilation, and 80-family-plan validation pass.
- Both corpus entry points reject absent freeze before sudo/container changes. Freeze creation rejects absent real smokes; test export rejects missing explicit unlock.
- Test log: `outputs/mvp_v5/review/contract_tests.txt`; final repeat: `outputs/mvp_v5/review/contract_tests_final.txt`; plan log: `outputs/mvp_v5/review/plan_validation.txt`.

Development failures were caught before capture: an invalid BPF expression, pandas boolean case conversion in a synthetic action fixture, and an unjustified bitwise CUDA assertion. They were corrected; none produced a real episode or evaluation score.

## Real smoke evidence and next action

The first `v5_smoke_001` attempt was interrupted by a user-confirmed manual shutdown after about 112 seconds. It is preserved as `_quarantine_v5_smoke_001_reboot_20260911T234648` (PCAP SHA-256 `28af0a56f6a61a80a9ef1d53e2925cc7d233292d5e5f84b5a362450cba30bf81`) and is not data. Entry points now self-enforce a blocking `shutdown:sleep:idle` inhibitor.

The two replacement smokes completed:

- benign legitimate SSH: `150.000136s`, 29 dense states, 184 observations, five-host sustained mixed traffic, no ATT&CK/LM truth;
- blocked scan/guess: `150.000197s`, 29 dense states, 228 observations, `T1046`, `T1110.001`, attempted `T1021.004`, and zero completed LM.

Both had zero packet drops, only inventory IPs, clean firewall/container shutdown, current source/image provenance, and passing raw/derived/V5 validators. In the blocked PCAP, the action was chosen before the five-second cutoff and applied afterward; the actor SYN receives a pivot TCP reset. Action and action-aligned-passive context/future/target arrays are exactly equal; only the former contains chosen action variables.

The capture-only freeze records 36 smoke artifact hashes and an 80-ID hash order. The corpus subsequently completed through the frozen generator. Model budgets, initialization treatment, thresholds, branch protocol, and metrics still require a separate train/validation freeze before sealed evaluation.
