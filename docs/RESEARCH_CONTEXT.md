# Research Context — Papers Read So Far

The purpose of this file is to preserve useful mechanisms and criticisms without prematurely locking the architecture.

---

# 1. Guo & Xie — Sensors 2025

"Research on Network Intrusion Detection Model Based on Hybrid Sampling and Deep Learning."

## What it does

TRBMA:

- TCN-ResNet branch;
- BiGRU branch;
- multi-head attention;
- current attack classification.

BS-OSS handles imbalance.

## Useful ideas

- causal/dilated TCN for real temporal receptive fields;
- parallel encoders only when branches have distinct semantics;
- class imbalance deserves explicit handling;
- scenario/trajectory-aware sampling;
- multi-scale fast/slow dynamics.

## Critical warning

The paper's input shape appears to use CIC feature columns as the sequence axis:

```text
(None, 78, 1)
```

That is not chronological network time.

Sequence-capable architecture does not automatically mean temporal network modeling.

---

# 2. Cao et al. — 2022 CNN + BiGRU IDS

## Useful ideas

- multi-statistic aggregation;
- redundancy-aware feature design;
- hard-example-aware imbalance handling;
- multi-branch modeling if each branch has meaningful semantics.

## Critical warning

Traffic vectors are reshaped to grayscale images for CNN processing.
Feature adjacency has no natural 2D spatial meaning.

If spatial structure is desired in this project, real network topology is preferable.

---

# 3. DeepStage — 2026

"Learning Autonomous Defense Policies Against Multi-Stage APT Campaigns."

## Relevant ideas

- POMDP framing;
- partial observability;
- provenance graph state;
- probabilistic current stage belief;
- CALDERA-driven attack episodes;
- cost-aware hierarchical defense actions.

## Critical distinction

DeepStage's PPO policy is model-free with respect to learned world dynamics.
It does not explicitly learn a neural transition model and roll it out.

Therefore it is highly relevant but not itself the final world-model formulation.

---

# 4. StageFinder — 2026

"Learning the APT Kill Chain: Temporal Reasoning over Provenance Data for Attack Stage Estimation."

## Architecture

```text
host + network provenance
   -> fused graph
   -> GNN
   -> graph embedding g_t
   -> LSTM
   -> current stage distribution
```

## Most useful detail for this project

Self-supervised pretraining uses:

### next graph-embedding prediction

\[
\hat g_{t+1}=W_oh_t+b_o
\]

\[
L_{pred}=\|g_{t+1}-\hat g_{t+1}\|^2
\]

### temporal contrastive loss

Encourages the history representation to match the true next graph more than negatives.

Then fine-tunes for stage labels.

This is a strong precedent for:

```text
large unlabeled telemetry -> dynamics pretraining
clean labeled episodes -> security semantic fine-tuning
```

## Limitation

Final deployed target is current-stage estimation, not full future rollout.

---

# 5. KillChainGraph — 2025

## Useful

- ATT&CK text embeddings;
- technique/phase semantic mapping;
- interpretable attack-path graph visualization;
- multi-model ensemble.

## Warning

Cross-phase edges are largely based on semantic similarity between descriptions.

Semantic relatedness is not the same as empirically learned temporal transition probability from observed network trajectories.

Use as knowledge/semantic prior, not as proof of world dynamics.

---

# 6. Mane & Rao — Explainable NIDS

## Methods discussed

- SHAP;
- LIME;
- ProtoDash;
- BRCG;
- CEM.

## Useful lesson

Different users need different explanations:

- model builder: global behavior/feature importance;
- analyst: local evidence and similar examples;
- operator/end user: local/contrastive explanation.

For this project, future explainability should also include graph/subgraph/edge evidence.

---

# 7. Ha & Schmidhuber — World Models (2018)

Core decomposition:

```text
V = observation encoder
M = predictive memory/dynamics
C = controller
```

The dynamics model learns a distribution:

\[
P(z_{t+1}\mid a_t,z_t,h_t)
\]

rather than merely predicting a label.

## Strongest transferable ideas

- compress high-dimensional observation into latent state;
- learn future latent dynamics;
- model stochasticity/probability;
- use learned world representation for downstream decisions;
- world-model exploitation/model error is a real problem;
- uncertainty matters.

For the current cyber project, defensive actions can initially be omitted:

\[
P(z_{t+1}\mid z_{1:t})
\]

and added later as:

\[
P(z_{t+1}\mid z_t,a_t)
\]

if counterfactual defense becomes part of scope.

---

# Combined current research direction

A coherent, still-provisional direction is:

```text
network graph S_t
   -> graph encoder E
   -> z_t
   -> temporal dynamics F
   -> probabilistic z_future
   -> future graph/network decoder
   -> MITRE + lateral movement interpretation
```

Possible StageFinder-inspired self-supervised objective:

\[
L_{ssl} = \lambda_{pred}L_{next-state} + \lambda_{ctr}L_{contrastive}
\]

Do not finalize architecture until the episode data and evaluation protocol are valid.
