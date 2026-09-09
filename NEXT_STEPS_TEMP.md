# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Verified branching checkpoint

Committed at `9533bf0`. The V3-train/validation-only outcome branch model is frozen for fresh evaluation. Do not reopen V3 test.

## Current batch: V4 role-balanced intervention-ready corpus contract

Add scenario support and a fully validated plan before asking for capture:

1. Extend `lab/run_episode.sh` without changing existing scenario behavior:
   - `matched_legitimate_ssh`: same network command/timing as direct credential movement, declared benign;
   - `credential_one_hop`: valid-credential LM without discovery/guessing;
   - `scan_guess_action_permit`: malicious prefix, recorded permit decision, successful SSH;
   - `scan_guess_action_block`: same prefix, recorded block decision, failed SSH attempt.
2. Add `defender_actions.csv` as separate action truth/conditioning metadata. It is never part of passive telemetry.
3. For the block scenario, install a source-specific TCP/22 REJECT rule inside the target container and remove it during cleanup. Add only the container capability/package required for this isolated lab action.
4. Record blocked SSH as `T1021.004` / `Lateral Movement Attempt`, not completed LM.
5. Add `configs/mvp_v4_episode_plan.csv` with 36 fresh episodes:
   - 12 role-balanced action-train episodes (six permit/block pairs);
   - 12 role-balanced sealed action-test episodes;
   - six sealed stopped/progressing passive episodes;
   - six sealed matched legitimate/direct-credential episodes.
6. Add `scripts/35_validate_v4_plan.py` checking fresh IDs/seeds, pair grouping, all six role permutations in action train/test, equal duration, semantic/action consistency, and immutable sealed assignments.
7. Add static shell/schema tests that do not require Docker/sudo/capture.
8. Document action observability: a chosen permit/block intervention may be a model conditioning variable only when it is known at forecast time; its future result is never input.

## Interactive boundary

Do not start V4 capture automatically. Capturing 36 × 120-second episodes requires the user to keep the laptop awake, rebuild/start the modified containers, and authorize one foreground sudo session.
