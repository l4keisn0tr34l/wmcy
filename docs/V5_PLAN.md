# V5 Time-Boxed Five-Host Plan

## Status

**Draft validated plan; not yet capture-frozen. Astra methodology review is required before capture.**

The purpose of V5 is to address the fresh V4 failures within a 3–4 day deadline. It does not retune any V3/V4 checkpoint or reuse an inspected test outcome.

Current draft hashes:

```text
configs/mvp_v5_episode_plan.csv
SHA-256 d4db26eff1eebfb50237c3a9d54bfe7eb9afa16966d1f256a15a62e10d69bade

configs/mvp_v5_split_assignments.csv
SHA-256 ea9c885cb6d2219369707effbccce167d80072c068f39b15f145eadc42ac8dd5
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

Allow approximately another 30–60 minutes for Docker transitions and processing. Capture remains resumable and may be paused only between episodes.

## Five-host inventory

```text
ws1     10.77.0.20
ws2     10.77.0.25
srv1    10.77.0.30
srv2    10.77.0.40
admin1  10.77.0.50
```

Each paired family uses a unique seed-derived ordering of actor, pivot, target, and two background hosts. Every physical host appears as actor, pivot, and target in each split. All activity stays inside the authorized `10.77.0.0/24` Docker bridge.

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

The two hosts not assigned actor/pivot/target generate background traffic. Paired alternatives share seed, roles, and background profile. Captures remain independent and therefore similar rather than packet-identical.

## Temporal schedule

V5 uses elapsed-time scheduling rather than accumulating command runtimes:

```text
capture start
~25 s: discovery where applicable
~50 s: guessing where applicable
~85 s: chosen action or SSH outcome
150 s: capture end
```

Actions align just after a five-second boundary. This leaves more than 60 seconds after intervention, preventing V4's missing sixth future-state failure. The action sequence still uses three complete pre-action states and six complete future states.

## Split policy

All paired-family members remain in one split. Seeds and ordered actor/pivot/target triples are disjoint across splits.

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

## Exit criteria before capture

- plan validator passes;
- five-host Docker health and isolation pass;
- one benign and one blocked-action smoke episode pass full processing;
- action timing leaves at least six complete future states;
- firewall cleanup is verified;
- background traffic is present according to profile;
- no V5 test file is loaded by a training script;
- Astra review accepts scope, splits, timing, and leakage rules.

## Remaining limitations

- only one five-host flat topology;
- small validation/test pair counts per scenario;
- deterministic SSH permit/block intervention;
- shared lab password and one service family;
- network telemetry still cannot reveal semantic intent for matched credential SSH;
- controlled lab evidence only.
