# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Planning assumption

Work toward a substantive 4–5 day improvement, while keeping a runnable checkpoint at the end of each day in case the deadline contracts. Do not prematurely switch the project to a one-day-only shortcut path.

## Current primary problem

The flattened RSSM forecasts aggregate state/LM well but fails host-relabeling equivariance. A shared final pair head could not repair upstream slot dependence. The next model must represent global, node, and directed-edge states structurally throughout encoding, recurrence, rollout, and decoding.

## Current code-writing batch: graph-equivariant RSSM foundation

Create a graph RSSM with these invariants:

1. Parse each 141-feature state as 15 global, 3×18 node, and 6×12 directed-edge features.
2. Fit preprocessing on training contexts only, sharing normalization statistics across all node slots and separately across all edge slots.
3. Use one node encoder for every host and one edge encoder for every directed pair.
4. Aggregate incoming/outgoing edge messages by sum/mean so host relabeling only relabels node states.
5. Maintain structured recurrent states:
   - invariant global hidden state;
   - equivariant per-node hidden states;
   - equivariant per-edge hidden states;
   - stochastic global latent with prior/posterior KL.
6. Use observation updates only during context/posterior inference; future rollout must use prior transitions without future observations.
7. Decode global, node, and edge future features with shared decoders.
8. Score future edge presence and LM source-target pairs with shared per-edge heads.
9. Pool graph features for invariant LM/ATT&CK heads.
10. Preserve the existing 15-second context, six-step/30-second rollout, whole-episode split, and target separation.

## Tests before full training

- exact input/output tensor shapes;
- future perturbation cannot change context posterior;
- deterministic host relabeling gives invariant LM/ATT&CK outputs and exactly permuted node/edge/pair outputs within numerical tolerance;
- tiny-batch loss decreases;
- no labels/future values enter `forecast(context)`;
- scaler means/scales are identical across corresponding node slots and edge slots;
- old RSSM remains loadable and unchanged.

## Training/evaluation

1. Train seeds 7/17/27 on V2 with KL 0.01/free-nats 0 as a supported initialization setting, selected on validation only.
2. Compare against Ridge, published flattened RSSM, and lower-KL flattened candidate.
3. Report state/active/quiet MAE, edge AP, LM/pre-LM, ATT&CK, pair AP/top-1, episode lead/false alerts, spread, and six-permutation equivariance.
4. Do not promote based on pair identity-order test alone.

## Public-data work after first graph checkpoint

1. Implement a host-rich UNSW-NB15 adapter using `srcip`, `dstip`, `Stime`, `Ltime`, and observable flow statistics; labels remain separate.
2. Partition by chronological capture ranges before sequence creation.
3. Use public telemetry for graph-dynamics pretraining and domain evaluation, not fabricated ATT&CK/LM truth.
4. Profile the one CICIDS2018 file retaining Src/Dst IP and the Friday PCAP as secondary dynamics sources.

## 4–5 day order

1. Graph RSSM foundation + exact equivariance tests.
2. V2 training and honest model comparison.
3. UNSW temporal adapter/dynamics pretraining plus several matched hard-negative lab episodes.
4. Fresh holdout evaluation and uncertainty/episode analysis.
5. Updated report/demo with fallback to the last verified checkpoint if the deadline moves forward.
