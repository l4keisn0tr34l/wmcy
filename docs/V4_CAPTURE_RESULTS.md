# V4 Capture and Action-Train Alignment

## Status

All 36 planned V4 episodes are captured, processed, and validated. No V4 test model prediction has occurred yet.

Two battery/interruption remnants (`lab_088` and `lab_090`) were quarantined under ignored `_quarantine_*` directories and regenerated from scratch. They are not present in any plan or model dataset.

## Corpus validation

```text
episodes:                    36
complete five-second states: 828 (23 per episode)
canonical observations:      2,559
ATT&CK action events:         90
chosen defender actions:      24
explicit zero-traffic states: 399
actual capture duration:      120.007–120.014 seconds
```

Scenario counts remain exactly as frozen in `configs/mvp_v4_episode_plan.csv`:

- action train: six permit/block seed pairs, all six role permutations;
- sealed action test: six permit/block seed pairs, all six role permutations;
- sealed passive test: three stopped/progressing pairs;
- sealed direct-credential test: three network-matched legitimate/malicious pairs.

`scripts/36_validate_v4_actions.py` passes 36/36 after fixing a pandas column-access bug (`metadata.pivot` collided with the DataFrame method; bracket access now reads the column). This was validator code failure, not episode-data failure.

All containers have only `INPUT ACCEPT`; no intervention firewall rule remains.

## Action timing

All 24 action records begin 0.207–0.212 seconds after a five-second boundary. Each action:

1. follows completed discovery/guessing;
2. is marked known at forecast time;
3. precedes the SSH success/blocked attempt;
4. falls inside the first six-step target window.

Permit episodes contain completed `T1021.004`/Lateral Movement. Block episodes contain `T1021.004`/Lateral Movement Attempt and no completed LM.

## Train-only action sequences

`scripts/42_build_v4_action_sequences.py` currently reads only the 12 action-training episodes and emits:

```text
context_states:                    [12, 3, 141]
action_type [permit,block]:         [12, 2]
action_pair:                        [12, 6]
future_states:                      [12, 6, 141]
future_edge_presence:               [12, 6, 6]
future_lateral_movement:            [12, 6]
future_techniques:                  [12, 6, 3]
future_lateral_edges:               [12, 6, 6]
lateral_movement_within_horizon:    [12]
```

The action types and LM outcomes are balanced 6/6. Each directed action pair appears twice, once permitted and once blocked. `--split test` fails unless `--unlock-test` is explicit; no test arrays have been created.

## Matched-prefix limitation

Separate packet captures are not numerically identical despite shared seeds. Using a scaler fitted only on the 12 training contexts, paired permit/block context normalized MAE is:

```text
mean: 0.164
range: 0.005–0.329
```

Mean paired future difference is 0.112. Thus seed matching controls roles/action schedule but does not create exact cloned telemetry. Action-conditioned evaluation must report this mismatch and must not claim a perfect randomized counterfactual trial.

## Leakage boundary

Model input may include:

- the three passive states ending at the intervention grid boundary;
- chosen action type;
- chosen directed source/target scope.

It may not include action outcome, post-action packets, completed/attempted LM truth, scenario, actor role truth, seed, or test-derived preprocessing. The action pair must be permuted consistently with graph host relabeling.
