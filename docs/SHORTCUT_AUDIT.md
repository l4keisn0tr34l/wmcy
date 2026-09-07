# MVP Dataset Shortcut and Leakage Audit

Date: 2026-09-07

## Scope

This audit asks whether the 20-episode controlled-lab corpus or the first PCA/Ridge world-model baseline can obtain apparently good results from fixed IP identity, capture timing, episode length, scenario templates, or ground-truth leakage instead of learning transferable telemetry dynamics.

Reproduce the quantitative checks with:

```bash
.venv/bin/python scripts/14_audit_mvp_shortcuts.py
```

Machine-readable output:

```text
outputs/mvp/model/shortcut_audit.json
```

## Executive conclusion

There is no evidence that a single common attack IP is the dominant predictor. Raw actor metadata alone has test AP 0.498, and fixed-slot presence masks have test AP 0.691 versus 0.683 for identity-invariant activity counts.

There are nevertheless two important issues:

1. **Critical dataset confound:** scenario-dependent capture length censors late negative windows. On test, every available sample at context state 7 or later is positive. A diagnostic using forbidden context-state index alone obtains AP 1.000. State index is not an input to the actual model, but this proves that the held-out sample distribution is unrealistically separable by episode phase/survival.
2. **Material fixed-slot sensitivity:** IP strings are not numeric inputs, but each IP always occupies the same node and edge slots. Consistently relabeling the three hosts changes LM probability by 0.234 on average and as much as 0.495 for one sample. Ranking metrics remain relatively stable, so this is sensitivity rather than proof that IP identity drives all performance.

The current corpus remains useful as a pipeline and end-to-end smoke test. Current held-out metrics must be called provisional and should not be the final judge evidence until fixed-duration episodes are generated.

## 1. Direct input-integrity result

Observed numeric input is `context_states` only. It contains 141 observable features over three five-second states.

No feature name contains:

```text
timestamp, time, state_id, episode, scenario, seed, actor, target,
technique, tactic, lateral, label, source_ip, destination_ip
```

The NPZ also stores future states and security targets, but they are separate arrays. Training and replay code pass only `context_states` into the forecast path. Replay joins future truth after prediction for evaluation/display.

### Preprocessing correction made during this audit

The original scaler and PCA used both training context and training future state rows when fitting preprocessing. No validation/test data was used, but this violated the project's stricter rule that preprocessing must be fitted on observable training contexts only.

`scripts/11_train_mvp_baseline.py` now fits `StandardScaler` and PCA exclusively on train `context_states`. Training future states remain supervised dynamics/decoder targets.

## 2. Fixed-IP and role findings

The tensor does not contain raw IP numbers. It does encode identity structurally:

```text
node_0 = 10.77.0.20
node_1 = 10.77.0.30
node_2 = 10.77.0.40
```

Directed edge slots are likewise fixed by IP pair.

### Role balance

All hosts appear in multiple roles over the complete corpus, but role balance is incomplete within scenario and split:

- both train benign-ping episodes use `ws1` as actor;
- train one-hop/two-hop progression actors are only `ws1` and `srv1`;
- no training lateral-movement event originates at `srv2`;
- only four of six possible directed LM pairs occur in training;
- test contains `srv2 -> ws1`, which is absent from LM training truth.

Training LM-pair counts:

```text
ws1  -> srv1: 1
ws1  -> srv2: 2
srv1 -> ws1:  1
srv1 -> srv2: 2
srv2 -> ws1:  0
srv2 -> srv1: 0
```

The selected `lab_023` replay's first pair is `srv1 -> srv2`, which occurs twice in training. The replay remains a valid held-out chronological example, but its pair success must not be presented as unseen-pair generalization.

### Identity diagnostics

Direct future-LM diagnostics on test:

```text
forbidden actor metadata only                    AP 0.498
slot-specific node activity + edge presence      AP 0.691
permutation-invariant activity/presence counts   AP 0.683
```

These results do not support the claim that one actor IP alone explains performance. The near-equality of the last two diagnostics says simple activity/topology presence is more important than which fixed slot carries it.

However, six equivalent host relabelings yield LM AP from 0.894 to 1.000 and per-sample score ranges averaging 0.234. The linear fixed-slot architecture is therefore not permutation invariant and can still learn role/IP-specific weights.

## 3. Critical episode-length censoring

Capture duration follows scenario execution time plus a fixed tail. More complex attacks therefore create longer captures:

```text
benign_ping mean states       9.7
legitimate_ssh mean states   11.0
scan_only mean states        13.3
failed_guessing mean states  13.7
one_hop mean states          18.3
two_hop mean states          20.3
```

A sequence requires three context states plus six complete future states. Short negative episodes stop contributing samples while progressing episodes continue.

On the test split:

```text
context states 2-6: available labels are negative
context states 7-14: available labels are positive
```

A state-index-only logistic diagnostic, which is intentionally forbidden and not part of the real model, obtains:

```text
test AP:            1.000
test pre-first AP:  1.000
```

The model does not receive state index or total episode duration, so this is not direct label leakage. It is still a severe evaluation confound: a real continuous benign network would continue producing late negative windows instead of ending when the scripted benign action ends.

## 4. Timing and scenario-template risk

All malicious progression follows one order:

```text
scan -> password guessing -> SSH lateral movement -> optional second hop
```

Action offsets vary, but only narrowly. Mean first-LM offsets are approximately 62.9 seconds for one-hop and 64.1 seconds for two-hop. Fixed SSH port, credentials, commands, topology, and tool implementations can produce repeatable byte/packet templates.

A diagnostic using only the final observable state obtains test AP 0.961; global context features without node identity obtain AP 0.982. Both exceed the current latent semantic head's original result. This means current LM success does not establish that temporal history, graph identity, or latent rollout is necessary for the semantic target. The world-model trajectory objective still beats persistence on future-state MAE, but the security interpretation task needs harder data and explicit no-history baselines.

## 5. Capture-order and environment-state risk

The Docker environment persists across episodes. ARP caches, SSH known-host files, and other container/network state can change over time. Test episodes are concentrated late in capture order (`lab_018`, `lab_020`, `lab_022`, `lab_023`), while validation is mostly in the middle. This can correlate environment drift with split even though timestamps and episode IDs are excluded from inputs.

No measured result proves that the model uses this drift. It remains a plausible confound that should be removed in the replacement corpus.

## 6. Ground-truth and temporal checks that passed

- All 20 planned episodes pass the integrity/leakage validator.
- ATT&CK and lateral truth are absent from observable state features.
- All 12 LM events coincide with the intended actor-to-target observable edge.
- State intervals remain dense, ordered, five-second windows.
- Context and future windows never cross episode boundaries.
- Splits are whole-episode, not random overlapping-window splits.
- Scaler/PCA fitting is now train-context-only.
- Hyperparameters and decision threshold are selected on validation, not test.
- Zero-duration ground-truth actions are aligned as points using exact UTC timestamps.

## 7. Required remediation before final claims

### Priority 0: replacement controlled corpus

1. Capture every scenario for the same fixed duration, recommended 120-150 seconds.
2. Continue benign/background telemetry through the complete duration so negative episodes supply late windows.
3. Broaden and randomize action start times independently of scenario where feasible.
4. Balance actor, pivot, target, and all six directed pairs within training rather than only over the whole corpus.
5. Reserve two evaluation modes:
   - balanced known-pair held-out episodes;
   - explicitly unseen-pair/role generalization episodes.
6. Interleave split capture order or reset/recreate containers between episodes.
7. Add harder negatives: scan-plus-no-progression, failed guessing followed by benign administration, and legitimate successful SSH with similar timing/volume.

### Priority 1: identity robustness

1. Apply consistent host-slot permutation augmentation to train contexts and futures.
2. Use deterministic but independent anonymized host slots for validation/test.
3. Prefer a shared-weight node/edge encoder or permutation-equivariant graph model over fixed slot-specific linear weights.
4. Repeat the six-permutation sensitivity test and require materially smaller probability ranges.

### Priority 2: evaluation

1. Always report persistence, last-state-only, global-only, and identity-mask diagnostics.
2. Report one episode-level first-alert result per episode in addition to correlated windows.
3. Evaluate strictly pre-first-LM samples and exact event lead time.
4. Do not select a favorable test replay without labeling it as selected.
5. Keep current metrics as smoke-test results, not enterprise or generalization evidence.

## 8. Corrected provisional model results

After context-only preprocessing:

```text
normalized future-state MAE         0.676
persistence normalized MAE          0.767
future-LM F1                        0.963
pre-first-LM F1                     0.957
future-edge AP                      0.412
LM-pair top-1                       0.429
```

These values verify that the end-to-end code still functions after leakage correction. They are not promoted as final evidence because the dataset-length and scenario-template confounds remain.
