# Project Context — Predictive Cyber-Defense World Model

## Challenge-level objective

Build an offline/open-source prototype that learns an evolving network/security state from telemetry and predicts attacker progression before compromise/progression completes.

Desired conceptual behavior:

```text
observed history S_(t-L+1:t)
        ↓
latent/belief state z_t
        ↓
learned dynamics
        ↓
future rollout z_(t+1:t+H)
        ↓
future network/graph state
        ↓
future ATT&CK behavior / lateral movement / risk
```

This is fundamentally different from classifying a current flow as malicious.

---

## Intrusion detection vs attack forecasting vs world modeling

### Intrusion detection

Typical mapping:

\[
x_t \rightarrow y_t
\]

Question:

> What is happening now?

Example:

```text
this flow -> PortScan
```

### Attack forecasting

Typical mapping:

\[
S_{t-L+1:t} \rightarrow Y_{t+1:t+H}
\]

Question:

> What malicious behavior is likely to happen in the future?

Example:

```text
past 60 s -> probability of lateral movement in next 30 s
```

### World model

Stronger formulation:

\[
S_t \rightarrow z_t
\]

\[
P(z_{t+1:t+H}\mid z_{1:t})
\]

with future state/graph decoding.

Question:

> How is the world itself likely to evolve, and what security consequences follow?

The project targets the third formulation, with forecasting heads built on top.

---

## Why graph structure matters

Lateral movement is relational.

A flat global vector may know:

```text
unique destinations increased
```

but not:

```text
ws1 suddenly contacted srv1, srv2, and db1
```

Therefore the current state representation preserves:

- network-wide state;
- each host's state;
- each directed host-to-host relationship.

The eventual model should be able to forecast not only "lateral movement is likely" but potentially:

\[
P(ws1 \rightarrow srv1 \text{ within } H)
\]

---

## Observation vs hidden security state

The defender never directly observes:

```text
attacker now owns credentials
```

The defender sees partial evidence:

- flows;
- packets;
- authentication/network patterns;
- topology changes;
- possibly host logs later.

This motivates a latent/belief-state formulation:

\[
o_t = \text{observable telemetry}
\]

\[
z_t = f(z_{t-1}, o_t)
\]

The learned `z_t` should summarize the security-relevant world sufficiently to predict the future.

---

## Current planned outputs

The final system should eventually expose:

1. Predicted future global/network state.
2. Predicted future host state.
3. Predicted future host-host edges.
4. Probability of future ATT&CK techniques/tactics.
5. Probability of lateral movement within multiple horizons.
6. Likely source/target host pair for movement.
7. Uncertainty.
8. Human-readable explanation grounded in observed/predicted features/subgraphs.

Example target demo:

```text
Observed history:
- new internal service discovery from workstation A
- rising internal fan-out
- repeated authentication pressure toward server B

Predicted future:
- new SSH edge A -> B in the next 30 seconds: 0.72
- T1021.004 / Lateral Movement: 0.68
- predicted edge novelty and port-22 activity are major drivers

If no such movement occurs, the probability should decay as new evidence arrives.
```

---

## Data strategy

### Controlled emulated episodes

Primary source of exact progression truth.

Advantages:

- exact attack action;
- exact start/end time;
- exact actor/target;
- exact ATT&CK technique/tactic;
- exact successful lateral-movement hop;
- packet capture retained.

### CICIDS2017

Useful but limited.

The uploaded TrafficLabelling CSVs:

- retain IP/port/flow features;
- have only minute-level timestamps;
- contain attack labels designed for IDS;
- do not provide clean exact lateral-movement trajectories.

Use primarily for broad dynamics/background work and verified behaviors.

### CSE-CIC-IDS2018

Candidate additional public dynamics source.
Needs adapter/schema inspection before use.

### UNSW-NB15

Candidate public/cross-domain source.
Use original temporal/host-rich files where possible, not only reduced ML tables.
Needs adapter/schema inspection.

### Future

- more precise PCAP/Zeek datasets;
- host telemetry;
- CALDERA-driven episodes;
- DARPA TC/OpTC-like provenance data if practical.

---

## Model-facing state concept

For time window `t`:

\[
S_t = (X_t^{global}, X_t^{node}, X_t^{edge}, G_t)
\]

### Global examples

- flow count;
- unique hosts/edges;
- new edge count;
- internal edge count;
- packets/bytes;
- SYN/RST/FIN counts;
- fan-out;
- port entropy.

### Node examples

- incoming/outgoing flows;
- incoming/outgoing bytes/packets;
- unique peers;
- new peers;
- unique destination ports;
- SYN/RST behavior;
- internal/external role.

### Edge examples

- flow count;
- bytes/packets;
- flags;
- duration;
- port diversity;
- internal edge;
- edge novelty.

These are observations, not ground-truth attack labels.

---

## Preliminary modeling direction — not a final architecture commitment

A first serious implementation may use:

```text
dynamic graph S_t
      ↓
graph encoder
      ↓
z_t
      ↓
temporal model over z-history
      ↓
predicted z-future
      ↓
state + edge + security heads
```

Possible temporal model families:

- GRU/LSTM;
- causal TCN;
- Transformer;
- state-space model.

Do not choose based on novelty alone.
Choose after the dataset representation and evaluation are valid.

---

## Primary loss concept

A future multi-task objective may look like:

\[
L =
\lambda_{state}L_{future-state}
+\lambda_{edge}L_{future-edge}
+\lambda_{mitre}L_{MITRE}
+\lambda_{LM}L_{lateral}
\]

Potentially also a contrastive/self-supervised term.

The **primary identity of the model remains future-state dynamics**, not attack classification.

---

## Why probabilistic futures matter

Attack progression is not deterministic.

The same observed history can lead to:

- failed credential attempts;
- attacker pausing;
- lateral movement to one of several targets;
- continued discovery;
- no attack progression.

The world model should eventually represent uncertainty rather than always outputting one deterministic future.

---

## Generalization goal

The system should not succeed by memorizing:

- IP address;
- fixed attack time;
- one playbook;
- one target host;
- one exact command;
- one dataset artifact.

Evaluation should include entire held-out scenarios/playbooks, and later cross-dataset/domain transfer.

---

## Key methodological warning

High IDS accuracy on CIC does not prove attack forecasting.

A world-model paper/result must demonstrate:

- true chronological prediction;
- future-state or future-event target;
- no overlapping train/test leakage;
- lead time;
- multi-step horizon behavior;
- trajectory/scenario generalization.
