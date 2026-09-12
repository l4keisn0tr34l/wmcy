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

**Status:** Accepted and implemented.

Keep PCA/Ridge as the interpretable benchmark. After V2 passes shortcut checks, implement a compact recurrent state-space model with a deterministic GRU state, diagonal-Gaussian stochastic prior/posterior, observable future-state/edge decoder, and future semantic heads. Accept RSSM only if it improves multi-step forecasting and uncertainty without increasing identity/timing shortcut sensitivity.

The senior's separate CICIDS2018 model is unavailable because its results were deemed too poor to continue. Its verbal description is motivation, not a verified baseline; this project implements and evaluates RSSM independently.

The compact V2 RSSM improves test state MAE from 0.354 Ridge to 0.280 and future-LM F1 from 0.645 to 0.800. It is retained as the preferred next candidate, not declared universally superior: edge AP falls from 0.230 to 0.222 and pair top-1 remains weak at 0.143.

---

## D018 — Report the RSSM training regime as hybrid

**Status:** Accepted.

Telemetry reconstruction, prior prediction, and KL dynamics losses are self-supervised. LM, ATT&CK, and pair losses jointly backpropagate through the same latent model, so the complete training regime is hybrid rather than purely self-supervised. A pure self-supervised/two-stage ablation is required before attributing downstream gains to unsupervised world-model learning alone.

---

## D019 — Test representation adaptation before model fusion

**Status:** Accepted experiment plan; not yet implemented.

On identical V2 splits, compare: current joint/unfrozen training, self-supervised pretraining with frozen downstream heads, self-supervised initialization followed by full fine-tuning, and zero-KL ablation. Report dynamics and semantic metrics separately. The senior independently reported that unfreezing helped their downstream classes, but their protocol is unavailable and remains an external hypothesis.

Do not add DANN without meaningful domain labels, and do not ensemble models until validation error correlation/oracle-gain analysis shows complementary signal. A marginal Ridge/RSSM edge-AP difference alone is not evidence for fusion.

---

## D020 — Retain KL model while tuning stochastic regularization

**Status:** Accepted after V2 ablation.

A zero-KL joint model improves several small-split point metrics (active-state MAE 0.984, edge AP 0.316, pre-first-LM F1 0.846, pair top-1 0.286), but mean rollout spread falls from 0.076 to 0.025 and worst-case host-relabeling LM score range is 0.814. Removing KL also abandons explicit posterior/prior alignment.

Therefore zero-KL is not promoted as the selected RSSM. Run a validation-only KL-weight/free-nats grid and require a useful forecasting/semantics/stochasticity trade-off. Never interpret lower Monte Carlo spread as better calibrated uncertainty.

---

## D021 — Use KL 0.01/free 0 only as the next research candidate

**Status:** Accepted after validation-only grid; not promoted as independent evidence.

Nine settings were screened without loading test, and the top two eligible settings were confirmed across seeds 7/17/27. KL 0.01/free-nats 0/seed 7 narrowly won the declared validation score and passed state, edge, LM, spread, and host-sensitivity gates. It improves the 20-rollout test point estimates to state MAE 0.276, edge AP 0.285, LM AP 0.926, and pair top-1 0.643 while retaining spread 0.068.

Use this checkpoint to initialize pair-decoder experiments, but retain the published RSSM as the stable headline. The selected score was close to KL 0.03, pair top-1 is only 9/14 windows, and the test split has been inspected repeatedly. Require permutation auditing and new episodes before treating the pair gain as robust.

---

## D022 — Reject fixed-slot pair top-1 as robust evidence

**Status:** Accepted after six-permutation audit.

Under equivalent host relabelings, the KL-tuned test pair top-1 ranges from 2/14 to 8/14 and only about 1.3% of non-identity top choices map back to the identity choice. The published model is also unstable. Therefore the tuned identity-order pair result fails the robustness gate and must not be a headline.

Implement a shared-weight source-target scorer next, while acknowledging that full equivariance requires replacing the flattened encoder/decoder with graph message passing.

---

## D023 — Do not adopt the isolated shared pair head

**Status:** Rejected after matched frozen-backbone experiment.

A shared `288 -> 64 -> 1` scorer over decoded source-node, destination-node, and edge features lowers test permutation-mean pair equivariance MAE from 0.031 to 0.021. However, permutation-mean pair AP falls from 0.267 to 0.232 and top-1 falls from 0.274 to 0.202; top-choice consistency remains near chance.

This indicates that slot dependence is already present in the flattened encoder/decoder. Do not add more complexity to the terminal head. The next architectural change must make node/edge representation learning permutation equivariant.

---

## D024 — Adopt graph RSSM as the dynamics foundation, not yet semantic winner

**Status:** Accepted first checkpoint.

A 358,115-parameter graph RSSM shares preprocessing, encoding, recurrence, decoding, edge scoring, and pair scoring across host/pair slots. It achieves deterministic and stochastic host-relabeling equivariance near numerical precision. V2 test state MAE improves to 0.252, active MAE to 0.895, edge AP to 0.394, and robust pair top-1 to 0.357.

Its initial LM F1/AP is only 0.476/0.738. Therefore adopt the architecture as the stronger world-dynamics foundation while retaining the flattened model as the current semantic benchmark. Train graph semantic heads separately before deciding on a combined successor.

---

## D025 — Stop V2-only semantic-head tuning after invariant readout

**Status:** Accepted.

Continuing the frozen pooled heads lowers test LM AP to 0.707. Reading decoded future global features plus permutation-invariant node/edge mean/max improves LM F1 from 0.476 to 0.667, but LM AP remains 0.744. Dynamics and exact equivariance remain intact.

The graph model's validation/test LM AP are both near 0.74, while the flattened model rises from validation AP 0.744 to test AP 0.910. Do not optimize repeatedly against this difference on an already inspected 14-positive-window test. Add matched hard negatives/progression episodes and public dynamics pretraining instead.

---

## D026 — Use UNSW only as grouped public telemetry pretraining

**Status:** Accepted after full local canonicalization.

UNSW original rows preserve host identities and Unix-second time but are heavily out of order. After sorting and one-hour gap segmentation, five source segments form only two connected capture groups because file times overlap. Never randomly split rows/segments across those groups.

Use observable UNSW flows for graph-dynamics pretraining and domain diagnostics. Keep raw categories separate and do not infer exact ATT&CK or lateral-movement progression from them. Missing TCP flag counts remain unavailable rather than fabricated.

---

## D027 — Retain public-pretrained graph RSSM as a transfer candidate, not unconditional replacement

**Status:** Accepted after validation-only public/lab selection.

UNSW observable-only pretraining improves lab-validation edge AP 0.361→0.430, LM AP 0.728→0.819, and exactly equivariant pair top-1 0.357→0.692. Diagnostic test point metrics also improve. But test false-alert episodes worsen from 2/4 to 3/4 and stochastic spread/error correlation falls from 0.722 to 0.199.

Therefore retain the checkpoint as evidence that public graph dynamics transfer, especially for edge/pair structure. Do not replace the stable headline model or claim calibrated uncertainty. Resolve the trade-off with matched hard negatives and a fresh sealed holdout, not further optimization on V2 test.

---

## D028 — Build V3 from seed-matched stopped/progressing prefixes

**Status:** Plan validated; capture pending interactive sudo.

Pair every `scan_guess_then_stop` episode with a same-seed `one_hop` episode. Both execute identical randomized discovery/guessing prefixes; only the progressing member performs successful SSH. Keep each pair within one split.

All 24 previously inspected V2 episodes become declared V3 development training. New episodes supply 12 train, 6 validation, and 6 sealed-test episodes. This prevents the repeatedly inspected V2 test from being presented as fresh evidence and directly tests whether the model distinguishes dangerous but stalled precursors from imminent LM.

---

## D029 — Reject public initialization as the V3 selection

**Status:** Accepted from prespecified V3 validation comparison.

Scratch wins the V3 validation joint objective (0.952 versus 1.060), state/edge metrics, and stochastic diagnostics. UNSW initialization has slightly higher fresh-test LM F1/AP, but worse fresh-test state MAE and edge AP. Do not select a model from that test difference. Retain scratch as the V3 comparison winner and public pretraining as a useful but distribution-sensitive V2 result.

---

## D030 — Treat matched-prefix LM as a branching-risk problem

**Status:** Accepted after fresh V3 paired audit.

Both models detect all 3 progressing test episodes but alert on all 3 stopped-prefix episodes. Scratch episode maximum probabilities differ by only 0.0027 on average within pairs. At the shared prefix, the future scenario-controller decision to execute successful SSH is not observable.

Do not claim a passive model can deterministically infer unobserved attacker intent. Separate dangerous-progression risk from exact eventual-outcome scoring, expose probabilistic branching, and collect action/intervention variables before counterfactual defense claims.

---

## D031 — Use CUDA for future neural experiments with prespecified larger batches

**Status:** Accepted after train-only hardware audit.

The RTX 3050/CUDA 12.8 path passes deterministic parity, finite gradients, and graph equivariance. Batch 128 is 1.78× faster per forward/backward step and uses only ~202 MB allocated VRAM, while batch 32 is slightly slower than CPU due launch overhead.

Use `--device auto` by default and prespecify larger batches for new experiments. Do not rerun or alter completed test results merely to benchmark GPU. Keep CPU fallback and save device-portable CPU checkpoint tensors. PCAP/data preparation remains CPU/disk-bound.

---

## D032 — Outcome-condition explicit branches; keep hidden intent unresolved

**Status:** Accepted for fresh-evaluation candidacy only.

Retain the explicit two-branch equivariant graph architecture and assign branches to no-LM/LM-within-30-second outcomes during training. Use context-only branch probabilities at inference and preserve both candidate graph futures. Do not reopen V3 test; evaluate only on fresh V4.

Reason: ordinary stochastic draws had little coverage/diversity, while an exchangeable trajectory mixture found generic telemetry modes but did not separate LM outcomes. Outcome conditioning produces materially different active graph futures and better validation proper scores while retaining causality/equivariance.

Guardrail: outcome truth is target-only, never model input. Near-0/1 conditional semantic heads are supervised construction, not discovered causal intent. All matched stopped and progressing validation episodes still alert, and validation Brier/ECE from 90 correlated windows are not calibration evidence.

---

## D033 — Treat chosen interventions as separate causal inputs in V4

**Status:** Accepted design; capture pending.

V4 records `permit_ssh`/`block_ssh` in `defender_actions.csv`, separate from passive telemetry and attack truth. An action-conditioned model may receive the action only after it is chosen and before any resulting packet. The passive model never receives it.

Reason: identical dangerous prefixes can diverge because of an explicit defender decision. Without the action variable, the passive process is genuinely multimodal; with it, counterfactual rollouts can ask what changes under permit versus block.

Guardrail: a blocked valid-credential SSH is `Lateral Movement Attempt`, not completed LM. Action train/test pairs use fresh disjoint seeds and both cover all six host-role permutations. The frozen passive branch checkpoint cannot be changed after V4 test capture is inspected.

---

## D034 — Partition precise PCAP by time bucket before aggregation

**Status:** Accepted and verified on full Friday capture.

Use fixed-width disk partitions selected by `bucket_index mod P`, aggregate each partition independently, then sort/k-way merge. Parse integer PCAP/PCAPNG timestamps directly rather than converting through float.

Reason: all packets for a directed five-tuple bucket land together even when capture records are out of order, while peak memory is bounded by one partition. Full Friday conversion processed 9,997,874 packets in 1m20.72s at 47,480 KB maximum RSS.

Guardrail: output is observable IPv4 packet-derived telemetry only. CIC labels remain separate; bucket duration is not whole-connection duration; Ethernet/IPv4 parser limitations are explicit.

---

## D035 — Keep Friday as one train-only connected capture

**Status:** Accepted.

Construct three-host induced graph samples with context-only rosters, five-second stride, and capture-causal edge novelty, but emit only `train.npz`. Do not random-split or temporal-split overlapping Friday windows and call them independent validation.

Reason: the available Friday file is one connected working-hours capture. Adjacent windows and host interactions are correlated. Fixed-setting dynamics pretraining can use it, but selection/generalization evidence must come from an independent development/evaluation source.

Guardrail: no CIC label is loaded; slot identity is anonymized; future activity never chooses roster members; frozen V3 test cannot select Friday pretraining.

---

## D036 — Pretrain Friday for a fixed epoch budget without internal selection

**Status:** Accepted initialization checkpoint.

Use seed 41001, CUDA batch 128, Adam 3e-4, and exactly 100 epochs on all Friday train-only sequences. Train reconstruction, future state, edge, and KL objectives; set every semantic weight to zero.

Reason: a random or temporal holdout from one overlapping connected capture would provide misleading selection evidence. A fixed protocol can produce an initialization while reserving transfer judgment for V4.

Guardrail: training-objective improvement is not validation. Do not fine-tune or evaluate this candidate against frozen V3 test, and do not claim Friday provides LM/ATT&CK semantics.

---

## D037 — Align V4 action samples at the pre-intervention grid boundary

**Status:** Accepted; train split built, test still sealed.

For each action episode, use the three complete five-second states ending at the boundary immediately before action enforcement. Supply chosen action type and directed scope separately, then target the boundary-containing state plus the next five states.

Reason: all recorded actions begin about 0.21 seconds after a five-second boundary, so passive context excludes action/result packets while the first target captures the intervention and SSH result.

Guardrail: action type/pair is known input; outcome and scenario truth are not. Test episode reading requires an explicit unlock after model settings freeze. Matched captures are not identical (train paired context normalized MAE mean 0.164), so causal claims must remain qualified.

---

## D038 — Carry scratch and Friday action models to V4 test without selection

**Status:** Frozen before V4 action-test access.

Train both ActionGraphRSSM initializations for exactly 400 epochs with seed 43001, full batch 12, and Adam 3e-4. Do not choose a winner from training fit; report both once on the sealed action-test pairs.

Reason: V4 has no independent action validation split, and training objectives cannot establish transfer. Carrying both models preserves the prespecified scratch-versus-public-initialization comparison without test-driven selection.

Guardrail: checkpoint hashes and evaluator metrics are fixed before creating test arrays. Report fixed 0.5 and frozen train-derived LM thresholds, graph-state effects as well as semantics, and paired-prefix mismatch.

---

## D039 — Exclude the complete `action_7002` pair for capture-boundary insufficiency

**Status:** Applied after explicit test unlock but before any model prediction.

The first test build found that `lab_090` had only five complete post-action states; the frozen action model requires six. Do not pad a partial window, shorten the horizon, or move the intervention backward. Exclude both `lab_089` and `lab_090` so paired permit/block analysis remains symmetric.

Reason: complete capture-bounded state targets are a causal temporal invariant. The exclusion is based solely on action/capture timing and was made before model results existed. The sealed cohort is reduced from six to five pairs and this protocol deviation must remain visible.


---

## D040 — Treat failed V4 passive alert transfer as a result, not a tuning target

**Status:** Adopted after one-shot frozen-checkpoint evaluation.

Do not adjust the V3 validation threshold or retrain the passive branch model against V4. Report that it alerted 3/3 stopped scan/guess episodes but 0/3 progressing episodes before a positive horizon, and missed 3/3 direct-credential progressions.

Reason: V4 is a fresh test corpus. Retuning would erase the generalization evidence. The state branches still show outcome specialization and oracle coverage, while the gate and precursor-specific alert behavior do not transfer reliably. Future improvement requires new train/validation episodes and broader scenarios/topologies.


---

## D041 — Time-box V5 around five-host background/action diversity

**Status:** Reviewed implementation; capture freeze awaits two real smoke episodes. No capture yet.

Use 80 paired 150-second episodes on one isolated five-host graph, with a real train/validation/test split, four balanced background profiles, scan and direct-credential actions, passive stopped/progressing pairs, benign controls, and a test-only matched-intent probe.

Reason: V4 showed that action conditioning works in the narrow lab but passive precursor warning and direct-credential generalization fail. V5 must address those failures within 3–4 days. Eighty captures require 3h20m of raw wall time and leave enough schedule margin for processing/training/reporting.

Guardrails: do not claim multiple-topology generalization; keep all paired alternatives in one split; keep intent probe out of training; schedule action by elapsed capture time and retain at least six complete future states; review, run smokes, then freeze before full corpus capture.

---

## D042 — Separate action decision from application and enforce smoke-backed V5 capture freeze

**Status:** Capture protocol frozen and all 80 episodes completed/validated; model protocol not started.

Choose the action before the forecast cutoff and apply it after; export its type/pair separately from three complete telemetry states. Use family-seeded timing variation, fixed hash capture order, sustained background from all hosts, and rotated validation/test cohort-profile assignments. Recreate the owned internal five-host network each episode, capture only inventory-to-inventory IPv4, and stop container-side descendants on every exit. Install raw completion metadata only after successful cleanup.

Reason: the draft had an after-cutoff decision timestamp, role/background and quiet-tail shortcuts, acquisition-order confounding, and no enforced freeze gate. These are data/observability problems, not problems to hide behind a larger neural model.

The new 345-feature exporter has separate passive, action, and action-aligned passive modes, strict schemas/reference/time/target validation, atomic immutable outputs, and explicit test locks. Fit no scaler during export. Future V5 candidates must share a train-context-only scaler; 141-feature checkpoint scalers cannot be reused. Shared parameter shape compatibility is verified, not beneficial transfer.

Freeze only after both non-corpus smoke episodes pass raw/derived/action/background/cleanup validation. Record source, image, and raw-smoke hashes in `configs/mvp_v5_capture_freeze.json`; refuse changed provenance rather than retrofitting it. Both corpus entry points enforce the gate. Capture freeze is not model/evaluation freeze. After the first smoke was interrupted by a user-confirmed manual shutdown through logind, both capture entry points were additionally made to self-enforce a blocking `shutdown:sleep:idle` inhibitor; the partial raw smoke was quarantined rather than completed or processed.

Limits: one flat topology, one executed SSH hop, synthetic background, residual small-cohort profile confounding, non-identical matched captures, and unidentifiable intent remain. Exact CPU causality passes; CUDA float32 reduction differences (~3e-8) require numerical tolerance rather than a false bitwise-equality claim. See `docs/V5_PRECAPTURE_REVIEW.md`.

---

## D043 — Accept equal-duration V5 corpus while disclosing residual paired timing phase

**Status:** Applied after capture, before model preprocessing/training or test export.

Accept all 80 planned episodes: each independently passes raw/derived/V5 validation, has a 150.000094–150.000615-second capture, exactly 29 complete dense states, and zero drops. Attack-intent and benign/legitimate mean raw durations differ by only 15 microseconds; maximum paired duration difference is 0.456 ms. This removes the historical outcome-dependent duration/censoring shortcut.

Retain and disclose that independent capture phases can place a paired forecast cutoff one state apart. This occurs in 2/20 action families; action-pair cutoff-offset differences are at most 3.3595 seconds (scan) and 2.7230 seconds (credential). Do not call the pairs packet-identical counterfactual twins. Each action sample nevertheless has three immediately preceding complete states and at least six complete futures, with minimum raw margin 42.294828 seconds.

Classify 12 post-block administrative SSH failures as real consequences of the chosen source-target firewall action and one tail timeout as bounded deadline behavior. Do not erase them: the defender action affects legitimate traffic on the selected pair as well as the focal attacker attempt. Quarantined partial attempts remain excluded.

This is all-split dataset integrity QA only. It must not choose preprocessing, architecture, hyperparameters, thresholds, or candidate models. See `docs/V5_CORPUS_AUDIT.md`.

---

## D044 — Freeze one deduplicated V5 train-context scaler and separate passive/action validation tracks

**Status:** Protocol implemented for pre-training freeze; V5 test remains sealed.

Fit one345-feature shared-slot scaler using exactly440 unique observable training context states, deduplicated by `(episode_id,state_index)` across passive and action train exports. Use no future, validation, or test values and never reuse the old141-feature checkpoint scalers. This preserves host equivariance, prevents overlapping passive windows from overweighting repeated states, and makes all normalized candidate errors comparable.

Compare scratch, fixed Friday, and frozen V4-scratch initialization under equal per-track budgets. The action track trains `ActionGraphRSSM` on24 action samples and selects one seed per initialization by validation state-plus-edge dynamics, not by the deterministic permit/block semantic label. The passive track trains an outcome-supervised two-branch `BranchingGraphRSSM` on336 passive windows and validates both ordinary passive forecasts and the eight action-aligned contexts without receiving action.

Use three prespecified seeds. Designate primary initialization(s) from validation before test while retaining one checkpoint per initialization for a prespecified transfer diagnostic. Primary LM threshold is0.5; validation-optimized F1 thresholds are secondary only. Oracle branch MAE is coverage only. Every retained model must pass finite-gradient, future-causality, all120-host CPU equivariance, and track-specific noncollapse gates.

Training completion does not unlock test. A separate evaluation freeze must hash selected checkpoints, scaler, evaluator, thresholds, and metrics before one explicit test export/evaluation. See `docs/V5_MODEL_PROTOCOL.md` and `configs/mvp_v5_training_protocol.json`.

---

## D045 — Supersede V5 protocol v1 pre-fit; require exact causality on CPU

**Status:** Applied before optimizer construction or any model/validation/test result.

The first V5 action invocation stopped in inherited V4 smoke code because a CUDA future-perturbation delta was `4.47e-8` rather than bitwise zero. Existing pre-training evidence already showed CPU delta0 and CUDA delta`2.98e-8`. Preserve the failed log and v1 freeze/scaler as invalid pre-fit artifacts; do not overwrite or pretend the run trained.

Protocol v1.1 runs exact future-causality and all120 deterministic equivariance gates on CPU copies of the exact initialized/trained candidates. Require CPU causality0 and equivariance below`1e-5`. Record CUDA deltas below`1e-6` only as numerical diagnostics. Continue actual optimization on CUDA.

This correction changes no data, scaler policy, initialization, architecture, weight, seed, budget, selection rule, threshold, metric, or test gate. It prevents a false bitwise-CUDA claim without relaxing structural causality.

V1.1 exposed a second stale inherited bound: Friday prefit CPU equivariance `2.62e-6` passed the declared `<1e-5` contract but failed helper code hardcoded at `<2e-6`. Archive that partial run, discard its temporary scratch states, and implement V1.2 with V5-owned checks reading the declared gate. Restart every action candidate from zero after a new freeze. See `docs/V5_PROTOCOL_INCIDENT.md`.

---

## D046 — Select scratch primaries on V5 validation and retain weak passive evidence

**Status:** Applied under frozen v1.2; test remains sealed.

Select scratch seed51003 as primary action candidate by the frozen dynamics score (`1.0192`, versus V4`1.0926`, Friday`1.2138`). Retain one checkpoint per initialization for the prespecified transfer diagnostic. Under the common scaler, scratch action validation state MAE is`.4727` versus persistence`.5658`; edge AP is`.3499` versus`.2989`. LM/pair metrics are perfect under deterministic permit/block, but factual state wins only7/8 contexts, so do not present semantics as complete graph forecasting.

Select scratch seed51003 epoch79 as primary passive branch candidate by the frozen composite (`.9318`, versus V4`.9446`, Friday`.9572`). V4 has slightly lower point state MAE (`.4554` versus scratch`.4609`) but does not win the frozen objective. Preserve the failure: scratch passive LM AP`.1615`, F1@.5`0`, pair AP`.0092`, and oracle gain only`.0012`; aligned-action contexts also have F1@.5`0`. Alternative branches add diversity but little useful coverage or warning.

All state comparisons use scaler SHA`3928af5...`. These are validation selection observations, not test/generalization. Freeze evaluator/checkpoint hashes and fixed thresholds separately before one test run. See `docs/V5_VALIDATION_RESULTS.md`.

---

## D047 — Atomically export/evaluate V5 test once after a separate review gate

**Status:** Implemented and validation-smoked; evaluation freeze intentionally not created pending critical review.

Freeze all six checkpoint hashes, scratch primary selections, shared scaler, validation-derived secondary thresholds, fixed0.5 primary threshold, exporter/evaluator/wrapper sources, sample counts, and metric list before test access. Include ATT&CK metrics and persistence references so deterministic action labels cannot dominate the report.

Export all three test modes into one temporary root, verify action/aligned common arrays bitwise, run the evaluator, hash artifacts, and atomically install one sealed result. If any post-access step fails, preserve the whole failed root and stop; do not silently delete evidence or tune/retry.

For the intent probe, evaluate the unique sample at each episode's metadata-frozen forecast cutoff and compare malicious direct-credential with matched legitimate SSH probabilities within each predefined family. Use only fixed0.5 and frozen validation thresholds. This measures observed discrimination, not hidden-intent recovery or packet-identical causality.

Do not create the evaluation unlock until the requested Astra-style final protocol review. See `docs/V5_EVALUATION_PROTOCOL.md`.

---

## D048 — Accept corrected V5 evaluation after full-lineage and permanent-attempt guards

**Status:** Astra review accepted2026-09-12 before any V5 test content access.

The initial implementation recorded hashes without enforcing all of them, allowed retries after failures/alternate output paths, and recomputed passive secondary thresholds in its CPU audit. Correct these before unlock: verify every frozen source/artifact/scaler/runtime; bind the freeze to a committed clean tree; permanently consume an exclusive fsynced run directory before export; permit only the claiming wrapper's child to consume one inference ticket; retain failure evidence and forbid automatic retry.

Use the original selected checkpoint thresholds exactly, not CPU-reoptimized values. Primary thresholds remain0.5. Retain existing model/seed/epoch/scaler/selection decisions unchanged. Save predictions and report profile/episode/per-horizon diagnostics plus primary passive warning lead to successful SSH action start. Such lead is not independently timed compromise completion.

Synthetic negative tests and CUDA validation reproduction pass. Intent cutoff selection is verified from generator code and eight synthetic fixtures, never real test episodes during review. Same-scaler action/passive comparisons remain descriptive because training cohorts differ; oracle branches are coverage only. See `docs/V5_FINAL_REVIEW.md`. Commit reviewed code and freeze before the one authorized test execution.

---

## D049 — Close V5 test after one successful evaluation; preserve passive warning failure

**Status:** Completed2026-09-12, freeze commit`e1497ff`, immutable `outputs/mvp_v5/sealed_test/` and permanent consumed claim.

Scratch remains primary in both tracks. Action state MAE`.459835` beats persistence`.539809`; factual chosen action gives lower state error8/8. Perfect LM/pair semantics are conditional on explicit chosen action/pair and deterministic lab outcomes, not general causal intelligence. Friday initialization does not improve scratch dynamics.

Passive scratch expected/oracle MAE`.472887/.470838` beats persistence`.538841` but provides only`.002049` candidate-coverage gain, LM F1@.5=0, and weak localization. Frozen low threshold detects3/6 progressing episodes while falsely alerting7/10 nonprogressing episodes. Preserve this failed warning tradeoff and the tiny intent probe's limits; do not rescue them with test-fitted thresholds.

Aligned action MAE`.459835` versus passive`.468434` is a small descriptive overall gain, not a pure conditioning ablation: cohorts/objectives differ and active-feature MAE favors passive`.661994` over action`.787490`. V4's lower passive point MAE does not replace the validation-selected scratch primary.

Verified hashes/lineage and point metrics from saved forecasts only. No post-test optimization or inference rerun is permitted. Update reports from immutable artifacts; future research needs new prospective data/protocols rather than reuse of V5 test for selection. See `docs/V5_TEST_RESULTS.md`.
