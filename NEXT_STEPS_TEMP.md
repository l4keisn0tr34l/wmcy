# V5 corpus complete — consistency audit and train/validation protocol next

## Observed corpus facts

- All80 planned episodes exist with raw+derived completion; general and V5 contract validators independently pass80/80. No active capture/container. Freeze SHA53069f98... still passes.
- Raw duration range150.000094–150.000615s; attack-intent vs benign/legit means differ ~15 microseconds. Every episode has29 dense complete 5s states (145s complete-grid coverage); paired max duration delta0.456ms and state delta0.
- Zero packet drops. Only planned metadata/splits.40 actions; action future margin min42.2948s. Event schedule ranges as frozen: T1046 20–28s, T1110.00143–49s, T1021.00482.034–107.906s.
- Background sustained from all5 hosts; expected profile kinds.13 failed commands:12 are post-effective block consequences on chosen source/target;1 is a bounded timeout at capture tail. Not uncontrolled capture failures.
- Residual paired mismatch: independent captures/absolute grid alignment yield action-pair forecast-offset differences max3.3595s scan and2.7230s credential; 2/20 action families differ by one forecast state index. Both retain immediate pre-action context and six complete futures. Disclose; not duration leakage.
- Quarantines excluded: historical V4 two, manually interrupted smoke, rejected first lab158 SSH-transport attempt. Replacement lab158 valid.

## Current batch

Create reproducible post-capture audit script/output; no model inference. Update CURRENT_STATE/TODO/MEMORY/V5 docs/decisions/output locations. Do not alter any capture-producing path enumerated by src/cyberwm/v5_integrity.py or rebuild images. Preserve capture freeze.

## Next research gate

Build TRAIN and VALIDATION exporters only. Before neural training, freeze one V5 train-context shared-slot scaler, candidate initialization rules (scratch/Friday/V4 shared parameters), budgets/seeds, passive/action-aligned comparison, branch treatment, thresholds, selection objective, metrics, and sealed-test unlock mechanism. Do not use V5 test arrays or outcomes for preprocessing/selection. Critical protocol merits Astra review; Sol may implement routine audited plumbing.
