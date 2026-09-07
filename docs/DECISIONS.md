# Architecture / Data Decisions

Use this as a durable decision log. Update decisions when evidence changes.

## D001 — The project is a world model, not an IDS

**Status:** Accepted.

Primary target is future network/security state dynamics.

Reason: the challenge requires attacker progression forecasting before compromise, not current-flow classification.

---

## D002 — Separate observations from ground truth

**Status:** Accepted.

Observable telemetry and scenario/ATT&CK truth live in separate files.

Reason: prevents direct target leakage and mirrors deployment.

---

## D003 — Use graph-structured network states

**Status:** Accepted as current representation.

Each time state contains:

- global network features;
- host/node state;
- directed communication-edge state.

Reason: lateral movement and attacker propagation are inherently relational.

---

## D004 — Use controlled emulation for clean progression truth

**Status:** Accepted.

The Docker lab produces packet capture plus exact ATT&CK action timeline.

Reason: common IDS datasets do not reliably provide exact successful lateral-movement trajectories.

---

## D005 — Keep public datasets

**Status:** Accepted.

CIC/UNSW/CSE-CIC are useful for dynamics/background/generalization but are not automatically trusted as progression truth.

---

## D006 — MITRE is an auxiliary semantic layer

**Status:** Accepted.

Model should predict future world dynamics and use ATT&CK heads to interpret them.

Reason: a label-transition model alone is not a world model.

---

## D007 — Use episode/scenario splits

**Status:** Accepted.

No random split of overlapping windows from the same timeline.

---

## D008 — Dense fixed-time states are required

**Status:** Accepted and implemented.

`03_build_graph_states.py` creates a dense fixed-time grid. Empty traffic windows receive explicit zero-valued global states; node and edge tables contain no rows for those empty states. This preserves the invariant that `state_id + 1` means exactly one configured time step later.

---

## D009 — Instantaneous ground-truth events are points

**Status:** Accepted and implemented.

`05_align_ground_truth.py` uses half-open state windows `[window_start, window_end)`. Positive-duration events use interval overlap; true zero-duration events align directly to the single containing state. Reversed events are rejected rather than silently repaired. No artificial timestamp precision or event duration is invented.

---

## D010 — Use explicit capture bounds

**Status:** Accepted and implemented for current generated episodes.

New episode metadata records capture start/end. State construction retains only complete fixed-time windows inside those bounds, including fully captured quiet tail windows. Canonical observations in partial boundary windows remain available for audit but are excluded from state aggregation and reported. Controlled traffic is aligned just after a five-second boundary. This prevents packet-derived boundaries or partial capture windows from silently distorting elapsed time.

---

## D011 — Validate before installing derived episode files

**Status:** Accepted and implemented.

The lab processor builds in a temporary directory, runs the integrity/leakage validator, and replaces derived outputs only after a pass. Raw PCAP, action truth, and episode metadata remain immutable.

---

## D012 — Keep generated data and environments out of Git

**Status:** Accepted.

Raw datasets, lab episodes, generated outputs, model artifacts, Python environments, and interpreter caches stay local. Git tracks reproducible code, configuration, and documentation.

---

## D013 — Final temporal architecture is not yet chosen

**Status:** Open.

For the time-limited MVP, start with an interpretable latent-transition baseline; only then consider a small recurrent/graph model. Do not treat any research paper's architecture as final.

---

## D014 — Fixed-shape MVP graph contract uses known-host masks

**Status:** Accepted for the controlled three-host MVP only.

Raw IP addresses select one of three stable known-inventory slots but are not scalar model features. Missing host and directed-pair rows are represented with zero activity plus explicit activity/presence masks. All six host-role permutations occur across training episodes, but LM-specific training covers only four directed pairs and no LM originates at `srv2`. Host-relabeling sensitivity is therefore an unresolved limitation. This is not yet a general enterprise graph contract.

---

## D015 — First baseline is a direct latent trajectory model

**Status:** Accepted and implemented as an MVP baseline.

Scaling and PCA fitted exclusively on observable training-context states encode the inputs; Ridge predicts six future latent states directly from three context latents; inverse PCA reconstructs future states; separate heads interpret predicted futures. Training future states are supervised targets, not preprocessing-fit data. This validates the end-to-end world-model objective quickly but is not an autoregressive probabilistic rollout and does not replace the planned neural graph dynamics model.

---

## D016 — Equal capture duration is mandatory for comparative episode evaluation

**Status:** Accepted and implemented.

The first 20-episode corpus used scenario execution time plus a fixed tail, causing progressing episodes to be longer than negative episodes. Complete-window sequence construction then censored late negative samples; a forbidden state-index-only diagnostic reached test AP 1.000. Future comparative corpora must capture every scenario for the same fixed duration, retain late benign/background windows, broaden timing, and balance LM directed pairs. Existing episodes remain immutable smoke-test data, but their semantic metrics are provisional.

V2 uses 120-second captures for `lab_025`-`lab_048`, balances all six training role permutations and directed LM pairs, and interleaves split capture order. All 24 episodes validate, have exactly 23 complete states, and contribute 15 sequences. The forbidden state-index diagnostic fell from AP 1.000 to 0.153, confirming that equal duration removed the dominant censoring shortcut.

---

## D017 — RSSM is the next candidate, not a substitute for data correction

**Status:** Accepted design direction; not yet implemented.

Keep PCA/Ridge as the interpretable benchmark. After V2 passes shortcut checks, implement a compact recurrent state-space model with a deterministic GRU state, diagonal-Gaussian stochastic prior/posterior, observable future-state/edge decoder, and future semantic heads. Accept RSSM only if it improves multi-step forecasting and uncertainty without increasing identity/timing shortcut sensitivity.

The senior's separate CICIDS2018 model is unavailable because its results were deemed too poor to continue. Its verbal description is motivation, not a verified baseline; this project will implement and evaluate RSSM independently.
