# V4 Role-Balanced and Action-Conditioned Plan

## Status

Plan and capture code are implemented. **All 36 V4 episodes are now captured, processed, and validated.** No V4 test model prediction has occurred. See `docs/V4_CAPTURE_RESULTS.md`.

Frozen passive branching checkpoint:

```text
models/mvp_v3_outcome_branching_graph_rssm_validation.pt
SHA-256 4f0524b5c25a9bd0e4f591362be948f0df12fa684b5140d55a7abcb147be874b
```

Frozen V4 plan:

```text
configs/mvp_v4_episode_plan.csv
SHA-256 6edfed04c25393ece9375a876fdf0a55a147bff400697128ab4e31b359ae811e

configs/mvp_v4_split_assignments.csv
SHA-256 2732c30f1f382346b28df8f6011b23f2ce5ec6ddcf0f4d35774e2c8f7fed219d
```

Changing either hash invalidates the sealed protocol and must be documented before capture.

## Questions

1. Does the frozen branch model generalize to fresh role permutations and timings?
2. Does it preserve plausible stopped/progressing alternatives on fresh matched prefixes?
3. Can passive telemetry distinguish a legitimate administrative SSH session from network-identical direct credential misuse? It should not be expected to recover intent that is absent from packets.
4. Does a known defender `permit_ssh`/`block_ssh` action explain diverging futures better than an action-unaware model?

## Corpus

Thirty-six fresh 120-second episodes, IDs `lab_073`–`lab_108`.

### Action conditioning: 24 episodes

Two alternatives share the discovery/guessing prefix and seed:

- `scan_guess_action_permit`: record permit; SSH succeeds; completed `T1021.004`/LM truth;
- `scan_guess_action_block`: install source-specific TCP/22 REJECT; SSH attempt fails; record `T1021.004` as **Lateral Movement Attempt**, not completed LM.

Six train pairs use all six actor/pivot/target permutations. Six disjoint sealed-test pairs also use all six permutations. The first action-conditioned model must use a fixed seed/epoch contract because this plan intentionally has no extra validation split; test remains unopened until its checkpoint is frozen.

### Passive branching: six sealed-test episodes

Three fresh seed-matched `scan_guess_then_stop`/`one_hop` pairs.

### Direct credential intent: six sealed-test episodes

Three fresh `matched_legitimate_ssh`/`credential_one_hop` pairs. Both execute the same network-visible SSH command with the same seeded timing. Their intent/truth differs. This is an observability control, not an invitation to force packet-only classification.

Across passive/direct test seeds, all six role permutations occur once.

## Action data contract

`defender_actions.csv` is separate from packet observations and attack truth:

```text
episode_id,start_time,end_time,action,source,target,known_at_forecast_time,details
```

The file is target/conditioning metadata. It is forbidden input to the passive model. An action-conditioned model may receive only a chosen action that is already known at the forecast cutoff.

Intervention scenarios align the action just after a five-second boundary. The action-conditioned context must end at or before `action.start_time`; no packets caused by the action or SSH result may enter context. The action one-hot may then condition future rollout. Future success/failure, LM truth, and post-action telemetry remain targets only.

## Why the block is real

The target container installs:

```text
iptables -I INPUT 1 -p tcp -s ACTOR_IP --dport 22 -j REJECT --reject-with tcp-reset
```

The rule is removed during cleanup, including error cleanup. Containers receive `NET_ADMIN` and the isolated image includes `iptables`. No host firewall is modified.

## Leakage controls

- opaque IDs;
- fresh seeds disjoint from V2/V3;
- whole seed pairs remain in one split;
- train/test action seeds are disjoint;
- both action splits cover all role permutations;
- equal capture duration;
- action truth separate from observations;
- blocked attempt is not mislabeled as completed LM;
- frozen branch checkpoint before capture;
- no test-driven threshold/model selection.

## Validation

Before capture:

```bash
.venv/bin/python scripts/35_validate_v4_plan.py
bash -n lab/run_episode.sh lab/generate_mvp_corpus.sh
docker compose -f lab/docker-compose.yml config >/dev/null
```

After capture/processing:

```bash
.venv/bin/python scripts/36_validate_v4_actions.py
```

This runs the general episode validator and checks action timing, source/target, scenario consistency, completed-versus-attempted LM semantics, and legitimate-session truth separation.

## Capture completion

The interactive capture completed. Two power-loss partial directories were quarantined and their episode IDs regenerated cleanly. `scripts/36_validate_v4_actions.py` passes 36/36. See `docs/V4_CAPTURE_RESULTS.md`.
