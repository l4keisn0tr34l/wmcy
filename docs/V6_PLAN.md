# V6 prospective dynamics, topology, and endpoint-evidence plan

## Status

**Design contract, deterministic episode plan, isolated seven-host Docker composition, authentication parser, and non-capture auth/policy feasibility smokes are implemented. No V6 PCAP episode, exporter, model, scaler, or test artifact exists. Nothing is frozen yet.**

Verified prospective plan:

```text
episodes:                    192
paired families:              96
train / validation / test:    88 / 40 / 64 episodes
family split:                 44 / 20 / 32
sealed test families:         16 in-domain + 16 topology-holdout
raw capture duration:          8.0 hours (before setup/processing)
episode duration:             150 seconds
maximum graph:                 7 hosts / 42 directed pairs
```

```text
configs/mvp_v6_episode_plan.csv
SHA-256 951eb14200740fb1da3e59300740cbd1ddc079817914b787a7863e87ce6930bc

configs/mvp_v6_split_assignments.csv
SHA-256 dd9a5f98e9fa337d4338e703b1182ed2ea0c8c1bdb1f30c82c07bd53e5fe652b
```

These hashes identify the current draft plan, not a capture freeze. V5 is closed and must not be used to select V6 models or thresholds.

## Why V6 exists

### Observed V5 facts

- Aggregate action/passive state dynamics beat persistence.
- Action-conditioned factual state error wins8/8 when permit/block and pair are already supplied.
- Passive completed-LM F1 is0 at the primary threshold.
- The validation-frozen low threshold detects3/6 progressing episodes and falsely alerts7/10 nonprogressing episodes.
- Passive oracle gain is only`.002049`; pair localization is1/36 top-1.
- One flat five-host topology and network-only telemetry cannot establish topology transfer or distinguish semantically different but network-similar SSH.

### Supported V6 design choices—not results

1. Add defender-observable authentication/session evidence rather than inventing hidden intent.
2. Separate static inventory/policy topology from dynamic telemetry and future-state loss.
3. Train action-present and action-masked models on the same intervention cohort/objective.
4. Reserve a topology/policy profile entirely for sealed test.
5. Evaluate ambiguous identical-prefix families as probabilistic branching/coverage tasks, not deterministic warning tasks.
6. Add second-hop progression so the target is attacker movement through the graph, not one isolated SSH event.

### Hypotheses to test

- **H1 dynamics:** masked topology-conditioned graph dynamics beat persistence on in-domain and held-out-topology state/edge forecasts.
- **H2 endpoint evidence:** network+authentication context improves validation-selected episode risk ranking and false-alert burden versus a network-only ablation on identical examples.
- **H3 action conditioning:** chosen action input improves future state/edge likelihood over an action-masked model trained with the same examples, seeds, budget, and losses.
- **H4 alternatives:** a proper finite-mixture future objective improves candidate coverage without sacrificing deployable mixture likelihood or collapsing to duplicate branches.
- **H5 progression:** context containing an observed first session plus pivot activity supports better second-hop edge/risk forecasts than network persistence and context-only security baselines.

None is an observed V6 result.

## Authorized isolated inventory

All proposed traffic remains on the existing authorized internal range`10.77.0.0/24`:

```text
ws1     10.77.0.20
ws2     10.77.0.25
ws3     10.77.0.27
srv1    10.77.0.30
srv2    10.77.0.40
admin1  10.77.0.50
jump1   10.77.0.60
```

No internet/public/unknown target is accepted. V6 uses a separate `cyberwm_v6` Docker project and `cyberwmv6` internal bridge; V5 and V6 cannot run concurrently because both reserve the same authorized subnet. A non-capture smoke proves the generated SSH/HTTP service rules on a smoke-only seed. It does not yet prove PCAP/state alignment or episode-level realization.

## Representation

### Dynamic observable context

Three chronological five-second states. Proposed dynamic state width is859:

```text
19 global features
7 × 24 node features
42 × 16 directed-edge features
```

V5 packet features retain their semantics. New proposed endpoint fields are only authentication records that a defender could observe at the target:

```text
auth success/failure counts
authentication method: password/public key
new remote authentication edge/source
per-host auth telemetry availability
```

`auth_success` is allowed only when its timestamp precedes the prediction cutoff. A focal future login remains a target, never context input. The runtime must record actual server-side log timestamps; scenario-controller timestamps cannot masquerade as observable endpoint telemetry. A separate session-open field is deliberately absent until a real smoke demonstrates a reliable observable record.

### Static defender-known conditioning

Static tensors are separate from the state target:

```text
node_inventory: [7]
edge_topology:  [42,5]
  inventory_pair
  same_zone
  ssh_allowed
  http_allowed
  route_hops_normalized
```

The decoder predicts dynamic future telemetry only. Static topology is excluded from state MAE so a model cannot improve headline error by copying unchanging masks. Binary topology values are not z-scored with traffic features.

### Deliberately excluded

- scenario/cohort/split/seed;
- actor, pivot, target, or fixed host-role identity;
- controller intent and future action;
- action result, permit/block consequence, or successful-LM truth;
- future packets, endpoint events, graph values, or edges;
- ATT&CK/LM labels;
- raw username as a memorization shortcut;
- scaler statistics fitted outside deduplicated training contexts.

## Topology/policy profiles

### Development profiles

- `flat5`: five seed-selected active hosts with open inventory connectivity; two padded inventory slots are explicitly masked.
- `segmented7`: seven active hosts assigned seed-randomized client/server/jump/admin zones with service policy features.

Both appear in train and validation. Host identities rotate across actor/pivot/target templates.

### Sealed holdout

- `dual_zone7_holdout`: seven active hosts in two seed-randomized zones connected through a jump host.

This profile appears only in16 sealed test families. Test also contains16 in-domain families, allowing transfer degradation to be separated from ordinary fresh-family error. Until real Docker routing/firewall behavior is implemented and packet-validated, this is a **planned policy/topology holdout**, not an achieved unseen-topology experiment.

## Cohorts

Each family contains two independent captures sharing seed, topology, service, background, and schedule. Independence means similar—not packet-identical.

| Cohort | Train/val/test families | Alternatives | Evaluation purpose |
|---|---:|---|---|
| Action ablation |10/4/8|same second-hop prefix; chosen permit vs block|same-cohort action-present vs action-masked dynamics|
| Informative passive |10/4/8|credential pressure/progression vs matched legitimate administration|whether observable auth evidence improves risk discrimination|
| Ambiguous passive |8/4/6|identical prefix then stop vs progress|mixture probability/coverage; **not** deterministic intent recovery|
| Two-hop progression |8/4/6|first session + pivot probe, then stop vs second hop|future pivot→target graph/LM prediction|
| Benign control |8/4/4|background only vs legitimate admin SSH|false-alert and background robustness|

Background profiles (`quiet`, `web`, `admin`, `mixed`) and authentication methods (`ssh_password`, `ssh_key`) are balanced by split and crossed within topology. The validator caught and rejected an initial service/topology confound before capture.

## Timing contract

Outcome-independent family schedules are generated in a separate RNG namespace:

```text
18–24s  discovery where applicable
40–47s  failed authentication pressure where applicable
80–84s  first-hop session for second-hop cohorts
89–94s  pivot evidence
96–106s decision
next absolute5s boundary: forecast cutoff
after cutoff: chosen action/focal future
150s capture end
```

A first-hop session is held for at least35s so its observable connection/session state remains in the three-state context. Worst-case cutoff plus30s forecast ends by145s. Runtime drift must reject the capture instead of shifting the scientific window.

## ATT&CK and completed-LM semantics

Candidate technique vocabulary:

```text
T1046      Network Service Discovery
T1110.001  Password Guessing
T1078      Valid Accounts
T1021.004  SSH
```

Generated ATT&CK truth comes from the intentional action plus server-side evidence. Password/public-key method is observable; malicious intent is not. A blocked SSH attempt may carry`T1021.004` attempt truth but is not completed LM. V6 should timestamp completed LM using server-observed accepted-authentication evidence and retain controller action timing separately for provenance.

## Planned model contract

Do not train until capture/export quality passes.

1. Fixed maximum seven slots with node inventory masks and42 directed pair slots.
2. Shared/equivariant node/edge operations.
3. Masked aggregation normalized by active allowed neighbors—not always by`node_count−1`.
4. Separate topology encoder/conditioning; dynamic decoder only.
5. Per-step state and communication-edge forecasts remain primary.
6. Per-step/horizon ATT&CK, completed-LM hazard, and directed next-hop heads interpret predicted futures.
7. Finite mixture with a proper likelihood/score, context-only weights, diversity diagnostics, and anti-collapse gates.
8. Action type/pair is separate and applied only after context when chosen before consequences.

### Required ablations

- network-only versus network+auth on identical rows;
- topology conditioning present versus masked on identical rows;
- action present versus action masked, with identical intervention examples/objectives/budgets;
- point dynamics versus finite mixture under one validation rule;
- scratch versus at most one frozen eligible initializer, only if schema-compatible.

V5 test metrics cannot determine architecture, loss weights, branch count, threshold, or initialization.

## Splits and evaluation

Whole paired families stay in one split. Seeds and topology-role triples are unique. Plan totals:

```text
train:       44 families /88 episodes
validation:  20 families /40 episodes
test:        32 families /64 episodes
             16 in-domain +16 dual-zone holdout
```

Primary metrics must include:

- dynamic state MAE/NLL by horizon, global/node/edge, active/quiet, and topology;
- edge AP/prevalence and next-hop pair AP/top-k;
- episode/family-balanced ATT&CK and completed-LM AP/Brier/log loss;
- thresholded detection, false-alert episodes, and lead to server-observed accepted authentication;
- deployable mixture score plus oracle coverage/diversity reported separately;
- action factual/opposite and action-present/masked comparisons;
- in-domain versus held-out-topology degradation;
- family bootstrap intervals where sample count permits.

Overlapping windows may optimize training but must not inflate confidence intervals or episode counts. Thresholds and model selection use validation only. Calibration claims require enough independent validation families and must be withheld if numerical support is inadequate.

## Build stages and fail-closed gates

### Stage0 — complete now

- deterministic plan and split generation;
- static topology/inventory contract;
- leakage/timing/balance validators;
- synthetic tests.

### Stage1 — runtime and telemetry feasibility

Current non-capture facts:

- one shared V6 image ran seven containers on internal `10.77.0.0/24` and stopped cleanly;
- actual Docker/OpenSSH UTC logs produced a password failure and public-key success; canonical events preserve host, remote host, method, result, and timestamp while discarding username/fingerprint;
- generated `dual_zone7_holdout` rules blocked direct cross-zone SSH/HTTP (`255`/`7`) while permitted zone→jump and jump→zone legs succeeded;
- generated `flat5` rules permitted active→active SSH/HTTP and blocked active→inactive access;
- these checks used smoke seed`999001`, which is absent from the prospective episode plan;
- two preliminary policy-smoke invocations failed closed after all traffic checks because the wrapper requested nonexistent profile`flat6`; both stopped all containers and wrote no passing artifact. The corrected contract name is`flat5`.

Evidence is under `outputs/mvp_v6/review/{auth_feasibility,policy_feasibility}.json`. This is engineering feasibility, not PCAP, forecasting, generalization, or model evidence. The two permitted dual-zone legs are not a claim that an L3 relay was traversed.

Remaining before any planned episode:

1. implement the full fail-closed V6 episode runtime around the isolated composition;
2. verify failed public-key authentication and per-host timestamped log collection;
3. verify server timestamps against PCAP and forecast cutoff;
4. verify inactive masks, policy tensors, and packet paths in processed states;
5. run topology×service benign/permit/block capture smokes;
6. quarantine every rejected capture smoke;
7. obtain independent review and freeze source/image/smoke hashes.

### Stage2 — development-only pilot

Capture/process a small train/validation-only subset. Audit endpoint missingness, topology realization, context availability, duration, drops, source identity balance, and same-prefix equality. Do not export or inspect V6 test.

### Stage3 — full capture

Requires AC power, interactive sudo, and an uninterrupted user-approved window. Eight hours is raw packet time; acquisition should be divided into resumable complete-episode blocks over multiple sessions. Never suspend or resume a partial capture.

### Stage4 — model freeze and one sealed evaluation

Fit preprocessing on deduplicated training contexts only, select on validation/family-balanced metrics, review/freeze evaluation, then access V6 test once. Permanent consumed-attempt semantics must be at least as strict as V5.

## Current limitations and unresolved engineering questions

- Docker realization of segmented/dual-zone policy is not implemented.
- Authentication log timestamp/parse fidelity is unverified.
- Seven-slot padding is variable active inventory, not arbitrary-size graph inference.
- SSH remains the only proposed remote-service family; password/key diversity is not service diversity.
-192 episodes are still synthetic and only32 independent test families.
- Exact same-prefix equality is a capture/runtime requirement not established by the CSV plan.
- Informative passive pairs intentionally contain observable differences; success would not imply hidden-intent recovery.
- The finite-mixture architecture/objective is not chosen yet.
- No V6 result exists.

These limitations are gates to resolve, not details to hide after evaluation.
