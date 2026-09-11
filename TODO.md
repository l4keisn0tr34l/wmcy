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
- [x] Add and evaluate valid-credential malicious SSH scenarios without prior scan.
- [x] Randomize action timing, baseline length, and attempt count from a recorded seed.
- [x] Randomize actor/pivot/target host roles.
- [x] Audit fixed-IP, role, timing, capture-length, and model shortcut risks (`docs/SHORTCUT_AUDIT.md`).
- [x] Enforce equal-duration 120-second captures and create the V2 replacement plan.
- [x] Balance all six directed LM pairs within V2 training; a separate unseen-pair test remains future work.
- [x] Capture and validate planned V2 episodes `lab_025`-`lab_048`.
- [x] Define a seeded matched-prefix hard negative and fresh V3 split plan.
- [x] Capture/process 24 paired V3 episodes and run prespecified scratch/public-init comparison.
- [x] Enable and validate CUDA execution on the local RTX 3050 without reopening V3 test.
- [x] Define validation-only branching-risk, exact-outcome, coverage, diversity, and calibration metrics.
- [x] Implement/train an explicit two-branch equivariant GraphRSSM without using V3 test.
- [x] Freeze the outcome-branch checkpoint and define the role-balanced/action-conditioned V4 plan.
- [x] Interactively rebuild the lab and capture/process `lab_073`–`lab_108`.
- [x] Validate all V4 action timing and completed-versus-attempted LM semantics.
- [x] Build V4 action-train-only intervention-aligned sequences with an explicit test lock.
- [x] Freeze scratch/Friday action-model settings, checkpoints, hashes, thresholds, and evaluator before V4 test access.
- [x] Build/unlock V4 action test once and run the frozen two-model evaluation; exclude one complete timing-invalid pair before prediction.
- [x] Evaluate the already-frozen passive branch model once on predetermined V4 passive/direct and pre-action rows; document failed early-warning transfer.
- [ ] Reset/recreate episode environment or interleave capture order to prevent cache/order drift.
- [ ] Add consistent host-permutation augmentation or a shared-weight graph encoder.
- [ ] Add richer benign background traffic.
- [x] Record episode metadata separately from observable telemetry.
- [x] Create a balanced 20-episode MVP capture plan and bulk runner.
- [x] Run and validate the planned 20-episode MVP corpus.
- [x] Produce 20 varied episodes for pipeline/model smoke tests; scale substantially after the MVP.

## DATA CONTRACT

- [ ] Finalize canonical event schema.
- [ ] Decide portable feature set shared across lab/public data.
- [x] Preserve raw data immutably in the current lab processing path.
- [x] Add a leakage-safe temporal canonical adapter for original host-rich UNSW-NB15 files.
- [x] Build context-only, anonymized UNSW induced-subgraph dynamics sequences.
- [x] Evaluate observable-only UNSW graph-dynamics pretraining and controlled semantic fine-tuning.
- [x] Canonicalize the full Friday PCAPNG with a disk-bounded, order-independent adapter.
- [x] Define/build causal context-only three-host induced graph sequences for Friday.
- [x] Pretrain Friday dynamics with fixed settings and no fabricated in-capture validation.
- [x] Evaluate Friday initialization under the frozen V4 action protocol; record that scratch won sealed state dynamics.
- [ ] Add CSE-CIC-IDS2018 adapter where useful.
- [ ] Version MITRE Enterprise ATT&CK STIX locally.
- [ ] Validate technique/tactic IDs automatically against that version.

## SPLITS

- [x] Create train/validation/test manifest at the **episode/scenario level**.
- [x] Ensure split happens before sequence generation.
- [ ] Hold out complete scenario/playbook variants.
- [ ] Add an unseen-path/unseen-implementation test split.

## FIRST WORLD MODEL

- [x] Create the MVP fixed-shape graph-state/sequence exporter with persistent known hosts and activity/presence masks.
- [x] Implement an interpretable PCA baseline state encoder producing `z_t`, with scaler/PCA fitted on train contexts only.
- [x] Implement a direct six-step Ridge latent trajectory baseline over `z_(t-L+1:t)`.
- [x] Implement a compact RSSM with recurrent deterministic and stochastic latent dynamics.
- [x] Add six-step stochastic prior rollout.
- [x] Implement a learned shared-weight graph RSSM encoder/decoder with exact host-relabeling equivariance.
- [x] Add baseline future-state reconstruction.
- [x] Evaluate baseline future-edge presence decoded from predicted future state.
- [x] Add Monte Carlo RSSM rollout spread and error-correlation diagnostics.
- [x] Run matched frozen/unfrozen and zero-KL representation-training ablations.
- [x] Tune KL weight/free nats with validation-only gates; retain lower-KL result as a candidate pending new-data validation.
- [ ] Calibrate RSSM predictive uncertainty.
- [x] Test frozen pooled and decoded-future invariant graph semantic readouts; rich readout improves F1 but data remains limiting.
- [ ] Add StageFinder-inspired next-step + contrastive pretraining if validated.

## SECURITY HEADS

- [x] Add baseline future ATT&CK technique multi-label interpretation from predicted future latent states.
- [ ] Add tactic head only if useful; do not force a linear kill chain.
- [x] Add baseline lateral-movement-within-horizon interpretation.
- [x] Add baseline source-target lateral-movement ranking and test a shared pair scorer; robust ranking remains unresolved upstream.
- [ ] Optional future compromise-state head.

## EVALUATION

- [x] Measure direct multi-horizon state error by global/node/edge feature group.
- [x] Measure RSSM open-loop rollout degradation vs horizon, including active/quiet states.
- [x] Measure baseline future-edge average precision/ranking.
- [x] Measure baseline MITRE multi-label metrics.
- [x] Measure sample-level LM recall and false positives.
- [x] Measure baseline LM source-target top-1; improve/qualify it.
- [x] Measure episode-level forecast lead time and alert behavior for Ridge and RSSM on validation/test episodes.
- [ ] Probability calibration.
- [ ] Unseen episode/playbook performance.
- [ ] Cross-dataset/domain-shift tests.
- [x] Run provisional global-only, last-state, identity-mask, state-index, and host-permutation shortcut diagnostics.
- [x] Audit RSSM pair/state/edge outputs across all six consistent host relabelings; reject fixed-slot pair score as robust.
- [x] Repeat state-index, actor, global-only, last-state, mask, and permutation diagnostics on equal-duration V2 data.
- [x] Compare pure self-supervised/two-stage RSSM against jointly supervised frozen/unfrozen training.

## EXPLAINABILITY / DEMO

- [ ] Feature/subgraph attribution for forecast.
- [x] Generate a self-contained held-out replay showing observed history and predicted/actual future state.
- [x] Show ranked future lateral host-host pair(s), while disclosing weak aggregate pair ranking.
- [x] Show ATT&CK interpretation derived from predicted future latent states.
- [x] Show model score, validation-selected threshold, forecast horizon, and non-calibration warning.
- [x] Generate a selected held-out V2 RSSM HTML/JSON replay with uncertainty and explicit selection disclosure.
- [x] Regenerate the standalone senior-facing HTML report with complete V4 action/passive results and inline diagrams.
- [ ] Visually inspect/polish and freeze the judge-facing HTML replay/report.
- [x] Draft and validate a time-boxed 80-episode, five-host V5 plan with balanced background profiles and a real validation split.
- [x] Implement isolated five-host Docker composition and V5 resumable runner; pass HTTP/SSH/firewall cleanup smoke checks.
- [x] Complete Astra pre-capture methodology/code review and correct timing, background, cleanup, and freeze gating.
- [x] Run, inspect, and validate two full V5 smokes; freeze capture/source/image/raw-smoke hashes. First attempt was manually interrupted and quarantined.
- [x] Capture/process and independently validate all V5 `lab_109`–`lab_188`; equal-duration/state audit passes 80/80.
- [x] Build five-host passive/action/aligned-passive contracts with test locks; 11 synthetic tests and CUDA/CPU causality/permutation checks pass.
- [ ] Freeze V5 scratch/Friday/V4-initialized comparisons on train/validation before opening test.
- [ ] Produce a judge-friendly example such as:
  "Observed internal discovery and credential pressure; model predicts a novel SSH edge from wsA to srvB within 30 s with 0.72 probability, interpreted as T1021.004 / Lateral Movement."

## LATER — only after predictive model is credible

- [x] Define V4 action records/splits for defensive-action conditioning `P(S_(t+1)|S_t,a_t)`.
- [x] Train/evaluate chosen-action-conditioned rollouts under a frozen V4 protocol; retain counterfactual caveats.
- [ ] Consider policy/control layer; do not let RL replace the world model.
