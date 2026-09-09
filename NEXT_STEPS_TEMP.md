# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Completed batch

Fixed-setting Friday dynamics pretraining is complete:

```text
source samples:                 5,775 train-only sequences
model:                          GraphRSSM, 358,115 parameters
protocol:                       seed 41001, CUDA batch128, Adam3e-4, 100 epochs
semantic loss weights:          exactly zero
training dynamics objective:    1.174 -> 0.350
future-state training loss:     0.905 -> 0.220
edge training loss:             1.076 -> 0.516
causality/equivariance delta:    0 / 4.47e-8
checkpoint SHA-256:             f8633c796f0523fc4537212ced3584d2f448ba55ac22be0cf94092a86e359ba0
```

This is training fit only, with no model selection and no validation/test loading. It is a candidate V4 initializer, not demonstrated transfer.

The standalone report includes this result at:

```text
/home/paprika/Downloads/rssm_eod_report.html
SHA-256 5887f19ff8472042759cd5b515d185eaf98b440ccf06ec1643b4aef4bac84c27
```

## Next blocking milestone: V4 capture

V4 code/plan is ready, but no V4 episodes exist. Capture requires the user to keep the laptop awake, rebuild/start the modified isolated containers, and authorize foreground sudo:

```bash
cd lab
docker compose down
docker compose up -d --build
./generate_mvp_corpus.sh ../configs/mvp_v4_episode_plan.csv
```

After capture:

1. Run `scripts/36_validate_v4_actions.py`.
2. Build V4 action-aligned contexts without opening action-test outcomes during development.
3. Freeze action-model architecture/seed/epoch/threshold settings using V4 action train only.
4. Compare scratch versus the frozen Friday initializer under that prespecified training protocol.
5. Open the 12 action-test and 12 passive/direct test episodes once.
6. Evaluate the already-frozen passive outcome-branch model on V4 test only.
7. Regenerate the standalone report, commit, and push.

Do not reopen or tune on V3 test.
