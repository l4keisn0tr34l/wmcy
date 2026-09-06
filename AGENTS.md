# AGENTS.md — Cyber World Model Project Instructions

## Read this first

This repository is for a **predictive cyber-defense world model**. It is **not** an IDS-classification project.

The central research objective is to learn the temporal dynamics of an enterprise network/security state:

\[
P(S_{t+1:t+H} \mid S_{t-L+1:t})
\]

The system should observe a history of network states, infer a latent security state, predict how that state will evolve, and then interpret predicted futures in terms of attacker progression, MITRE ATT&CK behavior, compromise risk, and lateral movement.

Do **not** silently turn this into:

```text
current flow -> benign/attack label
```

or:

```text
current flow -> attack family
```

Those are IDS tasks, not the project goal.

---

## Mandatory context-loading behavior

At the beginning of a new Pi session:

1. Read this file.
2. Read `MEMORY.md`.
3. Read:
   - `docs/PROJECT_CONTEXT.md`
   - `docs/CURRENT_STATE.md`
   - `docs/DATA_PIPELINE.md`
   - `docs/MITRE_MAPPING.md`
   - `docs/LEAKAGE_AND_SPLITS.md`
   - `docs/ROADMAP.md`
   - `docs/RESEARCH_CONTEXT.md`
   - `TODO.md`
4. Inspect the actual repository and generated outputs before assuming files exist or contain what the docs say.
5. If docs and files disagree, trust the files, report the mismatch, and update the docs only after verification.

Never answer from assumptions when the repository or an output file can be inspected directly.

---

## Project goal

Build an offline/open-source prototype that:

1. Represents network activity as chronological states, not shuffled flow rows.
2. Learns a latent representation of the evolving network/security world.
3. Learns predictive dynamics over that representation.
4. Performs multi-step future rollout.
5. Forecasts future graph/network changes.
6. Forecasts attacker progression and lateral movement before the relevant compromise completes.
7. Maps predicted future behavior to MITRE ATT&CK in an interpretable way.
8. Exposes uncertainty and useful explanations.

The intended conceptual pipeline is:

```text
raw telemetry
    ↓
canonical observable events
    ↓
dynamic graph state S_t
    ↓
state / graph encoder
    ↓
latent security state z_t
    ↓
temporal dynamics model
    ↓
predicted future latent states
z_hat_(t+1) ... z_hat_(t+H)
    ↓
 ┌──────────────┬─────────────────┬──────────────────┐
 ↓              ↓                 ↓
future state   future graph      security semantics
decoder        edge decoder      ATT&CK / LM / risk
```

---

## Non-goals

Do not make any of the following the primary system:

- a binary IDS classifier;
- a multiclass CIC attack classifier;
- a model that merely predicts the current ATT&CK stage;
- a Markov chain over ATT&CK labels with no telemetry dynamics;
- a semantic-similarity ATT&CK graph presented as learned network forecasting;
- an RL defense policy before the predictive dynamics model is working.

Simple classifiers may only be used as **diagnostics or auxiliary baselines**, and must never replace the world-model objective.

---

## Epistemic rules

Be strict about these categories:

- **Observed fact**: verified from current files, outputs, dataset documentation, or an experiment.
- **Supported design choice**: motivated by evidence/papers but not yet empirically proven here.
- **Hypothesis**: something worth testing.
- **Future idea**: not yet implemented.

Never state a hypothesis as if it were an observed result.

If a dataset does not contain a field, do not invent it.
If timestamps do not have enough precision, do not fabricate precision.
If an ATT&CK mapping is uncertain, write `UNKNOWN` or lower confidence rather than forcing a label.

---

## Explain every meaningful change

The user needs to understand this deeply enough to explain it to seniors and judges.

Whenever implementing or changing a pipeline stage, explain:

1. What goes in.
2. What transformation is performed.
3. What comes out.
4. Why the world model needs it.
5. What information is deliberately excluded.
6. What leakage or methodological failure the step prevents.
7. What limitation remains.

Do not give only an abstract block diagram when concrete code/files/commands can be discussed.

---

## Data-leakage rule

Model input may contain only information available to a defender at prediction time.

Ground truth is separate.

Never feed as observable input:

- CIC `Label`;
- `ground_truth.csv`;
- `state_ground_truth.csv`;
- ATT&CK technique IDs or tactics from ground truth;
- `has_lateral_movement`;
- future state values;
- future edges;
- future attack start/end times;
- scenario-controller actor/target truth;
- preprocessing statistics fitted using test/future data;
- fixed host/IP identities in a way that lets the model memorize attacker roles.

See `docs/LEAKAGE_AND_SPLITS.md`.

---

## MITRE rule

MITRE ATT&CK is a semantic/security interpretation layer, not the dynamics model.

For generated lab data, ATT&CK truth comes from the action the scenario intentionally executes.
For public datasets, mappings must come from documented scenario behavior plus ATT&CK definitions, with confidence/evidence recorded.

Do not infer ATT&CK truth by looking at a CIC attack label alone when the label is too broad.

ATT&CK is not a strict linear kill chain. Techniques can be multi-tactic, stages can repeat, branch, or overlap.

See `docs/MITRE_MAPPING.md`.

---

## Current primary data sources

1. Controlled Docker lab episodes: exact packet capture + exact ATT&CK action ground truth.
2. CICIDS2017 TrafficLabelling CSVs: useful public telemetry, but the uploaded copy has minute-level timestamps and imperfect progression labels.
3. CSE-CIC-IDS2018: candidate public dynamics source; integration not finished.
4. UNSW-NB15: candidate public/cross-domain source; integration not finished.
5. Future: additional precise PCAP/Zeek and host telemetry if needed.

Public IDS datasets are mainly for dynamics pretraining, benign/background diversity, and generalization tests unless their progression ground truth is verified.

---

## Current implementation status

The repository currently contains:

- CIC profiling;
- CIC canonicalization;
- PCAP-to-canonical conversion;
- graph-state construction;
- context/future sequence indexing;
- ATT&CK ground-truth alignment;
- a small isolated Docker lateral-movement episode.

The world-model neural network is **not implemented yet**.

Before model training, fix/verify the temporal state representation and episode dataset quality.

See `docs/CURRENT_STATE.md`.

---

## Coding / experimental workflow

Before editing code:

1. Inspect the relevant script.
2. Inspect representative input/output files.
3. State the intended invariant.
4. Make the smallest coherent change.
5. Run the relevant command/test.
6. Inspect the generated output rather than relying only on exit status.
7. Record durable decisions in `docs/DECISIONS.md`.
8. Update `docs/CURRENT_STATE.md` and `TODO.md` after meaningful milestones.

Prefer reproducible scripts over one-off notebook-only transformations.

Keep raw data immutable.

---

## Lab safety / scope

The current attack emulation is only for the isolated Docker subnet:

```text
10.77.0.0/24
```

Do not change scripts to scan, brute-force, or SSH into external/public/unknown targets.
Do not turn the lab generator into an internet attack tool.
All adversarial traffic generation should remain controlled and owned/authorized.

---

## Research attitude

The papers in `docs/RESEARCH_CONTEXT.md` are idea/evidence sources, not architecture commitments.

Important current research principles:

- real time must be represented as time;
- sequence-capable architecture does not automatically mean temporal modeling;
- graph structure is useful when it carries real network semantics;
- next-state/latent dynamics prediction is central;
- probabilistic futures and uncertainty matter;
- class imbalance is separate from world-model dynamics;
- trajectory/scenario-aware splits are mandatory;
- attention is not automatically explanation;
- evaluation must include forecasting/lead-time and unseen-scenario tests.

---

## If asked "what should we do next?"

Do not jump directly to a flashy neural model.

The immediate priority is:

1. verify/fix dense fixed-time state construction;
2. verify ATT&CK alignment, including short/zero-duration events;
3. regenerate `lab_001` with the current timestamp code;
4. build multiple varied benign and malicious episodes;
5. define episode-level train/validation/test splits;
6. only then implement the first latent dynamics model.

Read `docs/ROADMAP.md` for the complete sequence.
