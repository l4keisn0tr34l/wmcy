# V5 Time-Boxed Five-Host Plan

## Status

**Capture protocol frozen after two successful real smoke captures; full corpus not started. Model/evaluation protocol remains unfrozen.** See `docs/V5_PRECAPTURE_REVIEW.md` for corrections and smoke evidence.

The purpose of V5 is to address the fresh V4 failures within a 3–4 day deadline. It does not retune any V3/V4 checkpoint or reuse an inspected test outcome.

Frozen hashes:

```text
configs/mvp_v5_episode_plan.csv
SHA-256 b6676819c7c9cdf13e7a8b40b1861f209fb0a85310708028864d3de3b26a0946

configs/mvp_v5_split_assignments.csv
SHA-256 ea9c885cb6d2219369707effbccce167d80072c068f39b15f145eadc42ac8dd5

configs/mvp_v5_capture_freeze.json
SHA-256 53069f98ed2f8e686f973da6aa6f62f22f4d9b735def1da9868e7247783a20df
```

## Core questions

1. Can shared graph dynamics scale from three to five anonymous hosts?
2. Does action conditioning transfer across both scan/guess and direct-credential contexts?
3. Does varied legitimate background traffic reduce precursor-specific shortcuts?
4. Does a passive branch gate still fail on stopped/progressing and legitimate/malicious matched traffic?
5. Do scratch, Friday-public, and V4-lab initialization differ under one V5 validation protocol?

## Size and timing

```text
hosts:                 5
capture duration:      150 seconds per episode
episodes:              80
paired families:       40
raw capture wall time: 3 hours 20 minutes
train/validation/test: 40 / 16 / 24 episodes
```

Allow approximately another 30–60 minutes for Docker transitions and processing. Corpus generation is resumable only between complete episodes and may be paused only between episodes. Interrupted raw episodes are quarantined and recaptured from time zero. Smoke/corpus entry points self-enforce a blocking shutdown/sleep/idle inhibitor.

## Five-host inventory

```text
ws1     10.77.0.20
ws2     10.77.0.25
srv1    10.77.0.30
srv2    10.77.0.40
admin1  10.77.0.50
```

Each paired family uses a unique seed-derived host ordering; every physical host appears in each of its first three slots in every split. Only actor→pivot is an executed malicious hop: the legacy `target` and two `background_host` slots are audit metadata, not a second hop. All five hosts generate benign background. All activity stays inside the authorized `10.77.0.0/24` internal Docker bridge.

This is one larger flat topology, not evidence of unseen-topology generalization. It tests increased graph size and role/background diversity. Multiple topology families remain future work.

## Paired cohorts

### Scan action

```text
scan + guessing + chosen permit -> completed SSH movement
scan + guessing + chosen block  -> attempted but blocked SSH
```

### Direct-credential action

```text
chosen permit -> direct successful credential SSH
chosen block  -> direct attempted but blocked credential SSH
```

This directly tests the V4 direct-credential failure while keeping the defender action available before packet consequences.

### Passive prefix

```text
scan + guessing -> stop
scan + guessing -> successful SSH
```

No action or future intent enters model input.

### Intent observability probe

```text
network-matched legitimate credential SSH
credential-based malicious SSH with identical visible command
```

This cohort is test-only and is not used as semantic training supervision. It measures the known impossibility of recovering unobserved intent from equivalent network behavior.

### Benign controls

```text
background only
background + legitimate SSH
```

## Background profiles

Every split contains exactly balanced paired families from:

- `quiet`: sparse benign ping;
- `web`: repeated HTTP requests;
- `admin`: periodic legitimate administrative SSH;
- `mixed`: web, ping, and administrative traffic together.

All five hosts generate background through capture end−2 seconds, so source inactivity is not an actor-role marker. Paired alternatives share seed, roles, and background profile. Validation/test cohort profiles are rotated rather than repeating the old quiet/web versus admin/mixed partition. Two families per small cohort still cannot fully cross four profiles; stratify results and disclose residual confounding. Captures remain independent and therefore similar rather than packet-identical.

## Temporal schedule

V5 uses elapsed-time scheduling rather than accumulating command runtimes:

```text
capture start
20–28 s: discovery where applicable
43–50 s: guessing where applicable
80–103 s: chosen action decision (family-seeded variation)
next 5s boundary: forecast cutoff; application/SSH starts afterward
150 s: capture end
```

The action is chosen before the cutoff, but application begins about 0.2s afterward. Unexpected runtime drift or inadequate future margin rejects the episode rather than shifting/padding the window. The action sequence uses three complete pre-action states and six complete future states. Capture is measured from tcpdump readiness to stop request. Fixed hash ordering interleaves split/cohort/alternative acquisition rather than using CSV order.

## Split policy

All paired-family members remain in one split. Seeds and ordered first-three-host audit triples are disjoint across splits. This is not proof of independent effective interaction patterns: the executed actor→pivot pair can recur, and equivariant models deliberately ignore arbitrary host labels.

```text
train:      20 families / 40 episodes
validation:  8 families / 16 episodes
test:       12 families / 24 episodes
```

The test-only intent probe is never used for preprocessing, model selection, threshold selection, or training. V5 test remains sealed until model candidates and metrics are frozen using train/validation only.

## Model-facing contract

### Passive inputs

Three chronological five-second graph states containing only observable global, node, and directed-edge telemetry.

### Action-conditioned inputs

The same passive context plus separate chosen variables:

```text
action_type = permit_ssh | block_ssh
action_pair = directed anonymous host pair
```

The action pair must be permuted consistently with graph host relabeling.

The implemented exporter has passive, action, and action-aligned-without-action modes; raw context/future shapes are `[N,3,345]` and `[N,6,345]`, with 20 directed pairs. No normalization is fitted during export. See `docs/V5_PRECAPTURE_REVIEW.md`.

### Targets

Six future graph states, future edge presence, ATT&CK targets, completed-LM truth, and completed-LM pair truth. Attempted blocked SSH retains `T1021.004` attempt truth but is not marked completed LM.

### Deliberately excluded inputs

Scenario, cohort, seed, background profile, actor/pivot/target roles, ATT&CK truth, outcome, future packets, and absolute timestamps.

## Planned model comparison

Use one validation-frozen training protocol and carry eligible candidates to test:

1. scratch five-host ActionGraphRSSM;
2. fixed Friday observable-dynamics initialization;
3. frozen V4 action-model initialization transferred through shared equivariant parameters.

No candidate may be selected from train fit or V5 test. State metrics must use one V5 train-fitted scaler so candidates are directly comparable.

## Exit criteria before full corpus capture

- plan validator passes;
- five-host Docker health and isolation pass;
- one benign and one blocked-action smoke episode pass full processing;
- action timing leaves at least six complete future states;
- firewall cleanup is verified;
- background traffic is present according to profile;
- no V5 test file is loaded by a training script;
- Astra review accepts scope, splits, timing, and leakage rules;
- `scripts/49_freeze_v5_capture.py --freeze` records reviewed source/image/smoke hashes after both real smokes pass; both corpus entry points enforce the resulting freeze.

## Remaining limitations

- only one five-host flat topology;
- small validation/test pair counts per scenario;
- deterministic SSH permit/block intervention;
- shared lab password and one service family;
- network telemetry still cannot reveal semantic intent for matched credential SSH;
- controlled lab evidence only.
