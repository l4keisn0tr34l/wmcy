# V5 train/validation results

## Status

**Validation complete under frozen protocol v1.2; V5 test remains sealed and no test export exists.** These results select pre-test primary candidates. They are not sealed generalization evidence.

Independent recomputation:

```text
scripts/56_audit_v5_validation_models.py
outputs/mvp_v5/validation_audit/audit.json
status: PASS_VALIDATION_ONLY_TEST_SEALED
```

All six retained checkpoints reload with matching hashes and the shared scaler, have test-access markers false, exact CPU future causality0, and all120 CPU host permutations below`1e-5`.

## Action-conditioned validation

Frozen selection used group-balanced future-state MSE plus0.25 future-edge BCE. Semantic scores did not select the candidate.

| Initialization | Seed | Selection ↓ | State MAE ↓ | Active MAE ↓ | Edge AP ↑ | LM F1@.5 | LM Brier ↓ | Pair AP | Factual state wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Scratch** | 51003 | **1.0192** | **0.4727** | 0.7695 | 0.3499 | 1.000 | 0.000393 | 1.000 | 7/8 |
| Friday | 51003 | 1.2138 | 0.5373 | 0.7911 | **0.3575** | 1.000 | 0.000042 | 1.000 | 6/8 |
| V4 | 51003 | 1.0926 | 0.4987 | 0.7726 | 0.3400 | 1.000 | <0.000001 | 1.000 | 6/8 |
| Persistence | — | — | 0.5658 | — | 0.2989 | 0.000 | — | — | — |

Scratch is the frozen primary action candidate. All learned candidates beat persistence on state MAE and edge AP, but edge forecasting remains modest. The perfect LM/pair ranking is unsurprising because permit/block deterministically controls this one lab SSH outcome. It does not substitute for graph-state accuracy.

For the same contexts, mean permit-minus-block LM probability is0.985 scratch,0.993 Friday, and1.000 V4. Factual chosen action reduces mean state error versus the opposite action for all three, but not on every individual context. This is weaker than the V4 10/10 result and must be reported honestly.

## Passive alternative-future validation

Frozen branch selection combines trajectory/gate, edge, and semantic validation terms.

| Initialization | Seed/epoch | Selection ↓ | Expected MAE ↓ | Oracle MAE ↓ | Oracle gain | Diversity | Edge AP | LM AP | LM F1@.5 | Pair AP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Scratch** | 51003/79 | **0.9318** | 0.4609 | 0.4597 | 0.0012 | 0.0987 | 0.3935 | **0.1615** | 0.000 | **0.0092** |
| Friday | 51003/39 | 0.9572 | 0.4656 | 0.4644 | 0.0012 | **0.1160** | 0.3853 | 0.0651 | 0.000 | 0.0051 |
| V4 | 51002/114 | 0.9446 | **0.4554** | **0.4536** | 0.0018 | 0.0835 | **0.3955** | 0.0623 | 0.000 | 0.0045 |
| Persistence | — | — | 0.5340 | — | — | — | 0.3106 | — | 0.000 | — |

Scratch is primary under the frozen composite objective even though V4 has slightly lower point state MAE. All models beat persistence on state and edge validation metrics.

The security result is weak:

- fixed0.5 LM F1 is zero for every initialization;
- validation-optimized thresholds are very low (`0.058–0.100`) and achieve only F1`0.153–0.333`;
- LM AP is low, especially for transferred initializations;
- LM-pair AP is near zero;
- oracle state gain is only`0.0012–0.0018`, so the branches add diversity but little realized-future coverage.

On the eight action-aligned contexts without action input, all passive models again have fixed0.5 LM F1 zero. This supports the previously observed passive observability limitation rather than robust early warning.

## Same-scaler aligned comparison

Unlike the historical invalid cross-scaler comparison, both tracks use scaler SHA:

```text
3928af5e911006b713374c826e59162e68c7762d9d95ace15fa2b437ddf44be6
```

On the same eight action validation contexts:

```text
scratch action state MAE:            0.4727
scratch passive expected state MAE:  0.4539
persistence state MAE:               0.5658
```

The action model strongly predicts chosen LM semantics, but the broadly trained passive model has lower point state error on this tiny validation set. Therefore chosen action has not yet demonstrated a state-dynamics advantage in V5 validation. The sealed test must preserve both observations rather than report only perfect LM labels.

## Selected checkpoints

```text
models/mvp_v5_action_graph_rssm_scratch.pt
SHA-256 e080d969dd5c4676d0840a8749e46deda3674c324af2d849f4e8bc95ccd3f392

models/mvp_v5_action_graph_rssm_friday.pt
SHA-256 b81d5cfe1d08d562026e3715501d9be6f0128936520bfe5dd824e029e6bd5f03

models/mvp_v5_action_graph_rssm_v4.pt
SHA-256 d4b45ca146231f233594ce763f3e362e7dcebfb3bd3a111594b446e962f05501

models/mvp_v5_passive_branch_scratch.pt
SHA-256 d91a800bfdd2057e79f5c57ec0e9074e1dd598ae05d29dc88b6831ae99c84f1b

models/mvp_v5_passive_branch_friday.pt
SHA-256 5721b03ddad73fe938186616ed7cc950d0c9169656c1db0aaa509d0c1ff3ee03

models/mvp_v5_passive_branch_v4.pt
SHA-256 658398ebbcafd51a5a703d6ac9b7cac6ec4e76bc951c6c17f04d1a9218d1862b
```

## Claims and limitations

Supported validation observations:

- scratch is primary for both frozen selection objectives;
- learned dynamics beat persistence on these validation subsets;
- action conditioning captures controlled permit/block semantics but only modest edge dynamics;
- passive branches fail useful LM warning at0.5 and add little oracle coverage;
- Friday and V4 initialization do not win either frozen composite selection.

Not established:

- V5 test/generalization performance;
- robust intent recovery;
- unseen-topology or enterprise transfer;
- causal policy value;
- calibrated uncertainty.
