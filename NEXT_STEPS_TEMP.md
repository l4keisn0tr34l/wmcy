# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current EOD deliverable

Standalone HTML report:

```text
outputs/mvp_v2/report/rssm_eod_report.html
/home/paprika/Downloads/rssm_eod_report.html
```

Both copies have SHA-256:

```text
3aae31e51fd7988272b690ee2ffcba37e340d24315e5071d7e96f89a9d45fd84
```

The file is about 19.5 KB and contains all CSS, two SVG architecture/pipeline diagrams, charts, tables, metrics, selected replay, caveats, and next steps inline. HTMLParser found 539 elements and zero external `src`/`href` dependencies. The senior needs only this HTML file.

Reproduce with:

```bash
.venv/bin/python scripts/17_build_rssm_report.py
```

## Reported verified result

```text
                         RSSM    Ridge   persistence
state MAE               0.280    0.354      0.384
active-state MAE        1.031    1.089      1.137
future-LM F1            0.800    0.645
pre-first-LM F1         0.769    0.640
future-edge AP          0.222    0.230
LM-pair top-1           0.143    0.000
```

RSSM detects 2/2 progressing test episodes with mean exact lead 26.4 s; scan-only and failed guessing alert, benign ping and legitimate SSH do not. Selected `lab_048` replay: no prior LM, 87.8% ± 5.1%, exact LM 28.7 s later, correct top pair. Selection is disclosed.

## Files implemented

```text
requirements-rssm.txt
scripts/15_train_rssm.py
scripts/16_replay_rssm.py
scripts/17_build_rssm_report.py
docs/RSSM_RESULTS.md
```

## Next code batch after EOD review

1. Run pure self-supervised/two-stage RSSM versus current joint-loss ablation.
2. Improve edge AP and pair ranking using pair-wise/shared-weight decoding.
3. Add matched scan+guessing non-progression episodes to reduce false alerts.
4. Calibrate stochastic uncertainty.
5. Consider shared-weight graph encoding; defer actions until intervention data exists.

## Communication rules

- Send only `rssm_eod_report.html` unless source/evidence is requested.
- Never cite old confounded F1 0.963 as valid.
- Call complete RSSM training hybrid, not wholly self-supervised.
- Do not claim calibrated uncertainty, enterprise generalization, or action-conditioned counterfactuals.
