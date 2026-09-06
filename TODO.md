# TODO — Cyber World Model

## NOW — correctness before model training

- [x] Inspect the user's current local versions of `03_build_graph_states.py`, `05_align_ground_truth.py`, `06_pcap_to_canonical.py`, and `lab/run_episode.sh`. Do not assume the fixed zip was fully applied.
- [x] Make graph-state windows **dense**: every configured 5-second interval must exist, including zero-traffic windows.
- [x] Define zero-traffic global state values and empty node/edge tables without breaking state IDs.
- [x] Ensure state IDs correspond to fixed time increments, not merely "next non-empty state".
- [x] Make ATT&CK event alignment robust to instantaneous/very short events.
- [x] Generate a fresh episode (`lab_003`) using explicit nanosecond-formatted controller timestamps and UTC normalization; retain `lab_001` only as a legacy regression episode.
- [x] Verify BOTH T1021.004 hops appear in aligned truth for `lab_001` and fresh `lab_003`.
- [x] Add a validation script that checks:
  - monotonic timestamps;
  - exact fixed step between global states;
  - no duplicate state IDs;
  - every node/edge state references a valid state;
  - ground-truth event coverage;
  - no ground-truth columns in observations.
- [x] Record explicit capture bounds and retain only complete fixed-time windows.
- [x] Add atomic processing so invalid temporary outputs cannot replace prior derived files.
- [x] Retain canonical boundary observations for audit while excluding/reporting partial windows from graph states.
- [x] Align generated controlled traffic to a five-second boundary and enforce enough episode duration for MVP sequences.

## NEXT — build a real episode corpus

- [x] Convert `run_episode.sh` into parameterized scenario generation.
- [x] Implement benign-ping, legitimate-SSH, scan-only, failed-guessing, one-hop, and two-hop scenario branches.
- [ ] Add valid-credential malicious SSH movement without prior scan.
- [x] Randomize action timing, baseline length, and attempt count from a recorded seed.
- [x] Randomize actor/pivot/target host roles.
- [ ] Verify fixed-IP role shortcuts are controlled in the generated corpus and model loader.
- [ ] Add richer benign background traffic.
- [x] Record episode metadata separately from observable telemetry.
- [x] Create a balanced 20-episode MVP capture plan and bulk runner.
- [ ] Run and validate the planned MVP corpus.
- [ ] Produce at least 20–50 varied episodes for pipeline/model smoke tests, then scale substantially.

## DATA CONTRACT

- [ ] Finalize canonical event schema.
- [ ] Decide portable feature set shared across lab/public data.
- [x] Preserve raw data immutably in the current lab processing path.
- [ ] Add dataset adapters for CSE-CIC-IDS2018 and UNSW-NB15 where useful.
- [ ] Version MITRE Enterprise ATT&CK STIX locally.
- [ ] Validate technique/tactic IDs automatically against that version.

## SPLITS

- [ ] Create train/validation/test manifest at the **episode/scenario level**.
- [ ] Ensure split happens before sequence generation.
- [ ] Hold out complete scenario/playbook variants.
- [ ] Add an unseen-path/unseen-implementation test split.

## FIRST WORLD MODEL

- [ ] Create graph tensor loader.
- [ ] Implement a graph/state encoder producing `z_t`.
- [ ] Implement a temporal dynamics model over `z_(t-L+1:t)`.
- [ ] Predict next latent state first.
- [ ] Add multi-step rollout.
- [ ] Add future global-state decoder.
- [ ] Add future-edge decoder.
- [ ] Add uncertainty/probabilistic output where appropriate.
- [ ] Add StageFinder-inspired next-step + contrastive pretraining if validated.

## SECURITY HEADS

- [ ] Future ATT&CK technique multi-label head.
- [ ] Tactic head only where useful; do not force a linear kill chain.
- [ ] Lateral-movement-within-horizon head.
- [ ] Source-target lateral-movement edge head.
- [ ] Optional future compromise-state head.

## EVALUATION

- [ ] Next-state error by feature group.
- [ ] Rollout degradation vs horizon.
- [ ] Future-edge PR-AUC/ranking.
- [ ] MITRE multi-label metrics.
- [ ] LM event recall and false positives.
- [ ] LM target-host top-k.
- [ ] Forecast lead time.
- [ ] Probability calibration.
- [ ] Unseen episode/playbook performance.
- [ ] Cross-dataset/domain-shift tests.
- [ ] Ablations: global-only vs graph; no-history vs temporal; no-SSL vs SSL.

## EXPLAINABILITY / DEMO

- [ ] Feature/subgraph attribution for forecast.
- [ ] Show observed history and predicted future graph.
- [ ] Show likely new host-host edge(s).
- [ ] Show MITRE interpretation derived from predicted future.
- [ ] Show uncertainty and forecast horizon.
- [ ] Produce a judge-friendly example such as:
  "Observed internal discovery and credential pressure; model predicts a novel SSH edge from wsA to srvB within 30 s with 0.72 probability, interpreted as T1021.004 / Lateral Movement."

## LATER — only after predictive model is credible

- [ ] Consider defensive-action conditioning `P(S_(t+1)|S_t,a_t)`.
- [ ] Consider counterfactual defensive rollouts.
- [ ] Consider policy/control layer; do not let RL replace the world model.
