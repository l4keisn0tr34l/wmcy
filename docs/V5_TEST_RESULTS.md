# V5 sealed-test results — 2026-09-12

## Execution and audit

Astra reviewed and corrected the evaluator **before any test content access** (`docs/V5_FINAL_REVIEW.md`). Reviewed sources: `b1c5901`; inspected freeze commit: `e1497ff`. Script59 ran **exactly once**, from12:00:02 to12:00:09UTC, with status `SEALED_TEST_COMPLETE`. No sealed attempt failed. No post-test training, threshold selection, candidate selection, or inference rerun occurred.

```text
outputs/mvp_v5/sealed_test/{report.json,report.predictions.npz,provenance.json,execution.log}
outputs/mvp_v5/sealed_test/sequences/{action,passive,passive_action}/test/
outputs/mvp_v5/sealed_test_attempt/{claim.json,inference_started.json,complete.json}
```

The permanent claim is consumed and must remain. All12 sealed artifact hashes, six checkpoint identities, scaler/freeze/threshold lineage,54 finite forecast arrays, and nine point MAEs recalculated from **saved predictions only** were checked successfully. Read-only inspection snippets initially had report-key/scaler-call/reshape mistakes; correcting those did not invoke models or alter results.

SHA-256:

```text
evaluation freeze dfecbd58f04c81f762c1f84f9336c9ddaf16932cd460495fe099d2248554dc44
report            87688c91a10284f5cfa2424acbd3424d6af6a635e1444e3c86a2586aed580024
predictions       b06692a6bd748fbc5299019798f9f934a5a53df4ea8a3b1ca28b7d749328ab28
provenance        d616e633edc019403a912f8a44cc99fe7048cf5e5435b4ad0b4bb4ec9ef4819b
shared scaler     3928af5e911006b713374c826e59162e68c7762d9d95ace15fa2b437ddf44be6
```

## What was forecast

Three five-second observable graph states (15s, five hosts,345 features) produce six future graph states/communication edges (30s), plus horizon-level LM/ATT&CK/LM-pair probabilities. Action models also receive the defender-chosen pre-cutoff action and pair. Passive models do not. Truth, roles, scenario, future activity, and action outcomes are scoring metadata, never telemetry inputs.

All reported state errors use the same frozen train-context-only scaler. Scratch is the validation-selected primary for both tracks and remains so regardless of test comparisons.

Test scope: eight action contexts/eight episodes/four paired families;336 overlapping passive windows from16 episodes/eight paired families; eight action-aligned passive contexts. These are **not** hundreds of independent trials.

## Action-conditioned results

| Initialization | State MAE↓ | Active-feature MAE↓ | Edge AP↑ | LM F1@.5 | LM Brier↓ | Factual lower error |
|---|---:|---:|---:|---:|---:|---:|
| **Scratch primary** |**.459835**|.787490|**.411418**|1|.00002274|**8/8**|
| Friday |.472134|.779272|.383639|1|.00000166|5/8|
| V4 |.498926|.750962|.399944|1|.00000051|6/8|
| Persistence |.539809|.814268|.301626|—|—|—|

Scratch improves overall state MAE about14.8% over persistence. Edge prevalence is.295833; edge AP remains modest, not near-perfect graph reconstruction. Scratch factual-minus-opposite state error averages−.015762; mean permit-minus-block LM probability is.994532.

All action candidates have LM AP/pair AP/top1=1 on four completed-LM and four blocked contexts. **The action explicitly identifies permit/block and its directed pair; the lab outcome is deterministic.** Those semantic scores are not proof of sophisticated telemetry forecasting or general causal control.

Action ATT&CK micro AP is1, but every horizon contains attempted`T1021.004` and none contains the preceding scan/guess techniques. Per-technique AP is undefined for these single-class supports. Blocked SSH remains an attempted technique, not completed LM. This is not evidence of rich progression prediction.

Extreme validation-fitted secondary thresholds are brittle: scratch secondary F1=.857143, V4=.666667, despite primary F1=1. Do not tune them after test. Friday initialization did not improve scratch dynamics under this fixed budget.

## Passive branching results

| Initialization | Expected-point MAE↓ | Oracle MAE↓ | Diversity | Edge AP↑ | LM AP↑ | F1@.5 | Pair AP↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Scratch primary** |.472887|.470838|.098290|**.369933**|**.200029**|0|**.009435**|
| Friday |.474837|.472928|.116974|.360113|.092795|0|.008687|
| V4 |**.469090**|**.467163**|.083329|.368331|.123167|0|.007867|
| Persistence |.538841|—|—|.293380|—|—|—|

Scratch improves overall state MAE about12.2% over persistence. V4's slightly lower point MAE does not override the frozen scratch selection. Thirty-six of336 windows contain future completed LM (prevalence.107143); scratch predicts none positive at0.5. Pair top1 is1/36, versus V4's3/36 and Friday's0/36: localization remains weak.

Oracle gain is only.002049 for scratch (about0.43% of expected-point error). Diversity does not establish meaningful alternatives or calibrated uncertainty. On aligned action contexts, scratch oracle MAE `.468851` is actually slightly worse than the weighted-point `.468434`: a convex mixture can beat either candidate. Oracle is candidate coverage, not deployable accuracy.

Scratch ATT&CK AP: scan`T1046` .127029 (7 positive windows), password guessing`T1110.001` .565491 (34), SSH`T1021.004` .199625 (36). These overlapping-window diagnostics do not establish early unseen-scenario progression recognition.

## Aligned comparison: modest overall action gain, not a pure ablation

On the exact same eight contexts/targets/scaler:

- Scratch action state MAE `.459835`; scratch passive weighted-point `.468434` (about1.8% lower overall error for action).
- Action **active-feature** MAE `.787490` is worse than passive `.661994`.
- Edge AP: action `.411418`, passive `.413434`.
- LM F1@.5: action1, passive0.

Do not claim action improves every dynamics measure. Models were trained on different cohorts and objectives; this is a descriptive aligned comparison, not an experiment isolating conditioning itself.

## Warning and intent failures

For the primary passive model, across six progressing and ten nonprogressing episodes:

| Threshold | Progressing episodes detected within30s before first successful action | Nonprogressing episodes falsely alerted |
|---|---:|---:|
| Primary0.5 |0/6|0/10|
| Frozen validation `.1002940461` |3/6|7/10|

Secondary detected leads: `lab_176`15.200220s, `lab_178`20.200156s, `lab_180`10.200174s. These are leads to the controller-recorded **start of successful SSH**, not independently measured compromise completion. Secondary window precision=.209302, recall=.25, F1=.227848;34 false-positive windows. Lowering the threshold did not produce a useful warning/false-alarm tradeoff here.

The separately frozen **intent probe cutoff** covers four malicious/legitimate pairs (eight episodes):

- F1@.5=0; frozen-secondary F1=.4 (1/4 malicious detected,0/4 legitimate alerted at that particular cutoff).
- AP=.775, ROC-AUC=.625, mean malicious-minus-legitimate probability=.007997.
- Two pair differences are only about±.00005; independent capture variation is not controlled away.

Do not conflate the favorable-looking ranking or one specific cutoff with hidden-intent recovery. The broad episode warning test still falsely alerts on seven of ten nonprogressing episodes. The probe is tiny, independently captured, and not packet-identical.

## Background and uncertainty limits

Scratch state MAE by background profile:

| Profile | Action (2 contexts each) | Passive (84 windows each) |
|---|---:|---:|
| Quiet |.230166|.282140|
| Web |.506412|.466552|
| Mixed |.528933|.552226|
| Admin |.573830|.590628|

Balanced profiles expose difficulty differences but do not prove nuisance invariance. State groups/horizon curves and all episode/stratum scores remain in the immutable report.

Passive scratch Brier=.093954/ECE=.038004 on overlapping windows; intent Brier=.419770. These are small-sample diagnostics, not deployment calibration. The system is one synthetic flat five-host topology with one executed SSH hop. Enterprise/unseen-topology transfer, hidden intent, calibrated uncertainty, and causal policy value remain unsupported.

## Defensible conclusion

**V5 forecasts some controlled-lab graph dynamics better than persistence, and chosen actions recover deterministic permit/block consequences. Passive alternatives add little oracle coverage and fail useful completed-LM warning; modest graph accuracy and small-family limitations remain central.** Preserve these failures. Next use saved artifacts to update the standalone report/presentation, not to retune or rerun V5 test.
