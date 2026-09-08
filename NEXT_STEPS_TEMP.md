# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current verified checkpoint

Committed at `512488b`:

- 2,540,047 UNSW rows canonicalized without label leakage;
- 1,481 January train and 1,383 February validation graph samples;
- exact 141-feature lab-compatible layout;
- context-only roster selection and anonymized slots;
- two capture groups kept disjoint.

## Current batch: public dynamics pretraining and controlled-lab fine-tuning

Implement `scripts/28_pretrain_graph_rssm_unsw.py`.

1. Fit public normalization on UNSW January contexts only.
2. Pretrain one equivariant GraphRSSM on future state, context reconstruction, future edge presence, and KL only.
3. Select public checkpoint epoch on February public dynamics validation only.
4. Never load UNSW row ground truth.
5. Refit normalization on controlled-lab V2 training contexts only before transfer/fine-tuning.
6. Initialize graph dynamics weights from the selected public model; semantic heads begin effectively untrained and are learned from controlled lab truth.
7. Fine-tune matched seeds 7/17/27 on lab train, selecting seed/epoch on lab validation joint objective only.
8. Load the already-inspected lab test only after selection and report it diagnostically against the original graph RSSM.
9. Audit deterministic equivariance and save checkpoints/metrics.

## Gates

- public train/validation capture groups remain disjoint;
- public semantic loss weights exactly zero;
- no label-bearing public arrays exist or are loaded;
- causality and equivariance below 2e-6;
- public validation objective improves over initialization;
- lab test remains unavailable until fine-tune selection is frozen;
- output clearly reports whether transfer helps or hurts lab validation/dynamics/semantics.

## After this experiment

Do not repeatedly tune against V2 test. Preserve the result, then add matched scan-only/failed-guessing hard negatives and create a fresh sealed graph-model holdout.
