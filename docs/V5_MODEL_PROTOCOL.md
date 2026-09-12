# V5 train/validation model protocol

## Status

**Protocol v1.1 was also invalidated during temporary scratch training because an inherited helper imposed `<2e-6` instead of the declared `<1e-5`; v1.2 implements protocol-owned checks and awaits clean-tree freeze. V5 test remains sealed.** See `docs/V5_PROTOCOL_INCIDENT.md`.

The invalid v1 and v1.1 freezes/scalers/logs remain preserved. V1.2 changes no scientific threshold: its own smoke code enforces the already frozen CPU causality0, all120 equivariance `<1e-5`, and finite-gradient contract. No hyperparameter, data, split, scaler policy, or selection rule changed. Exact structural causality is tested on CPU; CUDA deltas below `1e-6` are numerical diagnostics. This remains the appropriate second critical methodology-review point. If Astra is unavailable, preserve the same pre-test review gate and do not weaken it after seeing validation results.

## Research question

Given three complete five-second observable graph states, forecast six future states (30 seconds), future directed communication, ATT&CK behavior, completed lateral movement, and its directed pair. Two distinct inference settings are evaluated:

1. **Passive:** infer alternative futures from telemetry only.
2. **Chosen-action conditioned:** receive a defender action selected before its packet consequences, then forecast the resulting future.

Neither setting is a current-flow IDS classifier.

## Data entering the protocol

| Track | Train | Validation | Input |
|---|---:|---:|---|
| passive two-branch | 336 windows / 16 episodes | 168 / 8 | `[3,345]` telemetry |
| action conditioned | 24 episodes | 8 | `[3,345]` telemetry + action type + directed pair |
| aligned passive diagnostic | — | 8 | exact same telemetry/targets as action validation, without action |

The action and aligned-passive common arrays and manifests are bitwise equal. Test arrays are absent. Whole paired families are disjoint across train, validation, and test.

## Shared train-only scaler

### Input

Only observable context states from `passive/train` and `action/train`.

### Transformation

Overlapping passive windows repeatedly contain the same state. Fitting directly on all context tensor rows would overweight middle states. The scaler therefore deduplicates by `(episode_id, state_index)`:

```text
16 passive episodes × 23 unique context-visible states = 368
24 action episodes × 3 context states                =  72
                                                        ---
unique observable train rows                          = 440
```

Global features receive one statistic each. Node features share statistics across all five anonymous node slots. Edge features share statistics across all 20 directed pair slots. This preserves host-permutation equivariance.

### Output

A 345-element mean and scale stored in:

```text
outputs/mvp_v5/model_protocol/shared_train_context_scaler.npz
```

### Why needed

All initialization regimes and both inference tracks must report normalized state errors under exactly one scaler. Otherwise a smaller MAE can be a normalization artifact, as happened in an earlier invalid cross-scaler comparison.

### Excluded information and prevented failures

No future value, validation value, test value, ATT&CK target, LM outcome, action result, scenario, role, seed, or absolute timestamp fits the scaler. Old 141-feature Friday/V4 scalers are never reused. This prevents future/test preprocessing leakage, fixed-slot statistics, and incomparable candidate errors.

### Remaining limitation

The scaler represents this synthetic V5 training distribution. It is not evidence of stable deployment normalization.

## Initialization comparison

Each track uses the same five-host architecture and V5 scaler under three prespecified regimes:

1. `scratch`;
2. fixed Friday observable-dynamics parameters (`f8633c...`);
3. frozen V4 scratch action checkpoint (`f69ece...`).

The Friday/V4 parameters use shared graph operations whose tensor shapes do not depend on three versus five node slots. Five-host node/pair index buffers are regenerated. Old scalers are discarded.

For the action track, Friday supplies 82 GraphRSSM tensors and leaves 11 action tensors seeded identically to scratch. V4 supplies all 93 compatible ActionGraphRSSM tensors. For the passive branch track, Friday and V4 each supply 82 base GraphRSSM tensors; nine branch tensors use the candidate seed. Action-only V4 tensors are deliberately ignored by the passive model.

Transfer is a hypothesis. Shape compatibility does not imply benefit.

## Action-conditioned track

### Input and transformation

`ActionGraphRSSM` observes three normalized states. It then applies one known action type (`permit_ssh` or `block_ssh`) and one consistently relabeled directed pair to the latent graph before six prior transitions.

Fixed budget per initialization/seed:

```text
seeds:        51001, 51002, 51003
epochs:       exactly 400
batch:        12
optimizer:    Adam 3e-4
gradient clip:10
augmentation: uniform random choice among all 120 host permutations
```

Loss weights retain the V4 contract: state1.0, reconstruction0.1, edge0.25, KL0.01, LM0.2, ATT&CK0.1, LM-pair0.1.

### Validation selection

For each initialization, choose one seed using deterministic validation:

```text
group-balanced future-state MSE + 0.25 × future-edge BCE
```

Semantic terms are trained and reported but cannot dominate seed/model selection through the deterministic permit/block label relationship. Ties within `1e-8` choose the smaller seed. Designate the best initialization by the same score before test; retain one frozen candidate per initialization for the prespecified diagnostic transfer comparison.

Primary classification threshold is fixed at0.5. A validation-best F1 threshold may be frozen as secondary diagnostics only and must not replace the primary result.

### Output

Six future graph states, six-step edge logits, six-step LM logits, future ATT&CK logits, and completed-LM pair logits. Counterfactual permit and block rollouts on the same context are diagnostic forecasts, not a learned causal policy guarantee.

## Passive alternative-future track

### Input and transformation

A two-branch `BranchingGraphRSSM` receives telemetry only. Supervised branch0 means no completed LM within30 seconds; branch1 means completed LM. The future outcome assigns branch responsibility during training, but the deployed branch gate sees context only.

Fixed budget:

```text
seeds:        51001, 51002, 51003
epochs:       exactly 250
batch:        128
optimizer:    Adam 3e-4
mixture temp: 0.25
gradient clip:10
```

The model retains the V3 outcome-branch loss and host-permutation augmentation. The best epoch and seed are selected solely by the frozen passive validation objective. One checkpoint per initialization is retained; one primary initialization is designated before test.

### Interpretation limits

Expected branch MAE is the context-gated forecast. Oracle MAE asks whether either generated branch covered the realized future and is **not deployable accuracy**. Brier/ECE on these small correlated samples are diagnostics, not calibrated deployment uncertainty.

The aligned validation set tests the same action contexts without giving the action. It measures the passive observability boundary; it is not a fair claim that a passive model should infer a hidden future intervention.

## Mandatory eligibility checks

Before any checkpoint can enter the later evaluation freeze:

- finite loss and gradients;
- exactly zero future-perturbation influence on context encoding/forecast;
- CPU deterministic host/action-pair equivariance delta below `1e-5`;
- action permit mean LM probability above block;
- nonzero passive branch diversity;
- source checkpoint, scaler, protocol, and trainer hashes match the freeze;
- no V5 test export exists during training.

A failed gate excludes the candidate; it does not justify changing the protocol after looking at test.

## Sealed evaluation gate

Training completion does not unlock test. A separate evaluation freeze must record:

- selected seed/epoch and checkpoint SHA for every retained candidate;
- primary action and passive candidates chosen from validation;
- frozen fixed/secondary thresholds;
- evaluator source hash and exact metrics;
- scaler and training-protocol hashes;
- explicit statement that test has not yet been exported or inferred.

Only then may a wrapper export V5 test once and execute one evaluator. No post-test retuning, threshold changes, candidate replacement, or rerun is allowed.

## Frozen metric families

- shared-scaler normalized state MAE: overall, global/node/edge groups, active, quiet;
- future edge AP;
- completed-LM AP, Brier, and F1 at0.5;
- completed-LM pair AP/top-1;
- factual action future versus opposite-action state error;
- passive expected/oracle branch state MAE, with oracle labeled coverage-only;
- test-only legitimate-versus-malicious matched-intent probe;
- scenario/profile diagnostics with explicit sample counts.

## Claim limits

Even a strong result remains evidence from one synthetic flat five-host topology, one SSH service/hop, deterministic permit/block actions, and small cohorts. It cannot establish enterprise readiness, unseen-topology generalization, learned defense-policy causality, intent recovery from indistinguishable prefixes, or deployment-calibrated uncertainty.
