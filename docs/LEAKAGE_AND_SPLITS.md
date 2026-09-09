# Leakage, Splits, and Causal Feature Rules

A forecasting model can appear excellent while learning information that would not exist at prediction time.

This document defines what is allowed.

---

# 1. Input rule

At prediction time `t`, the model may only use:

\[
\text{information observable at or before } t
\]

No future-derived value may enter the input.

---

# 2. Allowed model input

Current plan:

### Global state history

- flow count;
- unique hosts;
- unique edges;
- new edge count;
- internal edge count;
- bytes/packets;
- TCP flag statistics;
- fan-out;
- port entropy;
- other causally computed network aggregates.

### Node state history

- incoming/outgoing activity;
- peer counts;
- new peer counts;
- port diversity;
- packet/byte statistics;
- flag statistics;
- internal/external status.

### Edge state history

- source->destination activity;
- bytes/packets;
- flags;
- duration;
- port diversity;
- edge novelty;
- known topology if legitimately available.

### Temporal context

Past states only.

### Chosen defender action (action-conditioned models only)

A permit/block intervention may be supplied only if the defender has already chosen it at the forecast cutoff. It must be a separate action variable, not inferred by reading post-action packets. The passive model does not receive it.

---

# 3. Forbidden model input

Never feed:

```text
CIC Label
raw_cic_label
ATT&CK technique truth
ATT&CK tactic truth
ground_truth.csv
state_ground_truth.csv
has_lateral_movement
future state
future edge
future actor/target
attack start/end time
scenario name if it directly reveals attack family
future intervention result
post-action packets in a pre-action context
```

Do not feed absolute clock time if attack schedules are fixed and the model can memorize "attack starts at 15:04".

---

# 4. IP / hostname leakage

Current lab uses fixed:

```text
ws1 10.77.0.20
srv1 10.77.0.30
srv2 10.77.0.40
```

If every malicious episode always has ws1 as attacker, a model can learn identity rather than behavior.

Therefore:

- raw IP/hostname should be graph identity/index, not a scalar semantic feature;
- actor/target roles must vary across episodes;
- future dataset generation should randomize roles/topology or remap node IDs per episode;
- evaluation should hold out scenario variants.

---

# 5. Feature normalization leakage

If using z-score/min-max/imputation:

Bad:

```text
fit normalization on all episodes
then split
```

Good:

```text
split episodes
fit preprocessing on train only
apply fitted transform to val/test
```

The same applies to:

- imputation;
- feature selection;
- PCA;
- learned embeddings;
- calibration.

---

# 6. Sequence overlap leakage

Bad:

```text
build overlapping sequences over one episode
randomly split sequences
```

Example:

```text
train: S100-S112
test : S104-S116
```

The same incident appears in both.

Correct:

```text
split whole episodes/scenarios first
then build sequences separately inside each split
```

---

# 7. Public dataset split policy

Preferred axes:

- day/capture;
- attack session;
- scenario;
- attack family;
- playbook;
- cross-dataset.

Do not claim "unseen attack generalization" from a random held-out subset of flows from the same execution.

---

# 8. Novelty features must be causal

Example:

```text
is_new_edge
```

is valid if it means:

> this source->destination pair has not appeared before the current state.

It must not use future knowledge such as:

> this edge will only appear once in the entire episode.

Any historical baseline feature must be computed from past/current data only.

---

# 9. Dense time matters

A missing traffic window is not permission to skip time.

For a 5-second model:

```text
13:34:40
13:34:45
13:34:50
```

must exist as three consecutive states even if the middle window is empty.

Otherwise the model cannot know whether one state transition took 5 seconds or 50 seconds.

---

# 10. Ground truth is allowed as a target

It is correct to train:

\[
z_t \rightarrow \text{future MITRE labels}
\]

or:

\[
z_t \rightarrow \text{future LM target}
\]

provided those labels do not appear in the input.

Similarly, future states are the primary world-model target.

---

# 11. Past model predictions

A future architecture could feed its **own past belief/prediction** recurrently.

That is different from feeding ground-truth ATT&CK labels.

Do not implement this until explicitly designed and evaluated.

---

# 12. Reporting rule

Every metric/result must state:

- split unit;
- forecast horizon;
- context length;
- time resolution;
- whether attack family/playbook was seen in training;
- whether public/lab domains differ;
- how preprocessing was fit.

This prevents misleading "99%" results.
