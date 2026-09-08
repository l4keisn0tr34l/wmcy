# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Verified checkpoint

Public graph pretraining is complete at commit `89e0980`:

```text
validation: state MAE 0.260, edge AP 0.430, LM AP/F1 0.819/0.813, pair 9/13
V2 diagnostic test: state/active MAE 0.251/0.877, edge AP 0.405,
                    LM AP/F1 0.785/0.778, pair 10/14 under every relabeling
```

Trade-off: 3/4 negative test episodes alert and spread/error correlation falls to 0.199. Report SHA-256 is `f7d9cb52f04fc0f056cf46806756efc058d89306ae132d425267a45218703d4e`.

## Current batch: matched hard-negative data and fresh holdout

1. Add controlled `scan_guess_then_stop` scenario to `lab/run_episode.sh`.
2. It executes the exact discovery + failed-password prefix of `one_hop`, with the same seeded role/timing machinery, but deliberately performs no successful SSH lateral movement.
3. Preserve T1046/T1110.001 truth while LM truth remains zero; this is not benign traffic.
4. Add a 24-episode paired plan: each seed has one stopped precursor and one one-hop progression episode.
5. Assign six seed pairs to train, three to validation, and three to a fresh sealed test; keep all six role permutations in the new training pairs.
6. Define V3 manifests so all previously inspected V2 episodes become development training, while only newly generated episodes populate V3 validation/test.
7. Add a generation script/command with resume behavior through the existing corpus runner.
8. Run shell/config validation and document exact split semantics.

## Interactive blocker

Host packet capture needs `sudo -v`, and the sudo timestamp is currently unavailable. After code/config validation, ask the user once to run the foreground generation command. Docker can be started without sudo if needed. Do not fabricate episode outputs before capture.

## After generation

Process/validate all new episodes, build V3 manifests/sequences, train the public-pretrained graph RSSM using V3 train/validation only, and open fresh V3 test once. Compare false alerts, robust pair ranking, state/edge forecasts, and stochastic diagnostics.
