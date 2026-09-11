# V5 reviewed implementation — blocked on two real smoke captures

## Completed

- Review corrections: independent seed-matched nuisance timing; hash acquisition order; all-host background through end−2s; rotated validation/test cohort profiles; internal inventory-only network capture; fresh owned containers; bounded commands and fail-closed cleanup/provenance.
- Action chosen before forecast cutoff, physical application afterward; insufficient horizons/drift rejected.
- Five-host exporters (script47): passive/action/passive_action; 345 features / 20 pairs; raw [N,3,345]→[N,6,345]; no scaler fitting, no truth/roles in inputs, explicit test lock, immutable atomic output.
- Script48: 11 synthetic tests PASS, all120 CUDA permutations max3.35e-8 in final repeat, exact CPU future perturbation0, CUDA2.98e-8 within tolerance, finite gradients. Actual Friday/V4 parameter shape loading passes; no old scaler reuse or prior-test inference.
- Script49 plus both corpus entry points enforce capture freeze backed by real smoke/source/image hashes. Missing freeze/missing smoke/test-lock rejection verified before capture side effects.
- Updated docs/V5_PRECAPTURE_REVIEW.md, plan, decisions D042, pipeline, commands, current state, TODO, MEMORY.

## Immediate next action (interactive user terminal)

sudo -n still reports password required. No real smoke or V5 corpus capture exists. Do not bypass sudo using another capture mechanism.

```bash
cd /home/paprika/Documents/153/wm
systemd-inhibit --what=sleep:idle --mode=block bash lab/smoke_v5.sh
```

AC power required. Runs ONLY v5_smoke_001 legitimate SSH and v5_smoke_002 blocked scan/guess, each150s, then processing/QA/export. Does not freeze or start corpus. Existing running V5 containers are the earlier configuration; runtime recreates only this owned project using reviewed configuration. On failure preserve partial raw directory for quarantine, never overwrite or suspend mid-capture.

## After both smokes pass

Inspect raw timing, metadata, truth/action, sustained background logs, zero capture drops, cleanup receipt, graph/state/target alignment and exported shapes. Then explicitly run:

```bash
.venv/bin/python scripts/49_freeze_v5_capture.py --freeze
```

Only afterward may full corpus begin, using frozen hash order. No V3/V4 reruns/tuning; no sealed V5 model evaluation. Capture freeze is NOT model protocol freeze. V5 model budgets, common train-only scaler, source initialization treatment, passive branch comparison, thresholds, and evaluation metrics still need train/validation-only definition/freeze. One flat topology and one executed hop remain the scope; role audit triples do not establish topology independence.

Current plan SHA b6676819c7c9cdf13e7a8b40b1861f209fb0a85310708028864d3de3b26a0946; split SHA ea9c885cb6d2219369707effbccce167d80072c068f39b15f145eadc42ac8dd5. No capture-freeze JSON exists yet. Remote CSE download untouched. Sol is suitable for routine smoke/corpus execution; next critical model/evaluation freeze merits Astra review.
