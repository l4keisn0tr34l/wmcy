# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## V5 draft ready for Astra review — do not capture yet

A time-boxed V5 draft is implemented and statically validated:

```text
80 episodes / 40 paired families
150 seconds each / 3h20m raw capture
5 hosts / 345-feature expected graph contract
train 40 / validation 16 / sealed test 24 episodes
```

Cohorts: scan permit/block, direct-credential permit/block, passive stopped/progressing, benign controls, and test-only legitimate/malicious intent probe. Every split has complete five-host actor/pivot/target coverage; each split balances quiet/web/admin/mixed background families exactly. All 40 ordered role triples are unique across splits.

Files:

```text
configs/mvp_v5_episode_plan.csv
configs/mvp_v5_split_assignments.csv
scripts/46_validate_v5_plan.py
lab/docker-compose-v5.yml
lab/run_v5_episode.sh
lab/generate_v5_corpus.sh
docs/V5_PLAN.md
```

Draft plan hashes (not frozen):

```text
d4db26eff1eebfb50237c3a9d54bfe7eb9afa16966d1f256a15a62e10d69bade
ea9c885cb6d2219369707effbccce167d80072c068f39b15f145eadc42ac8dd5
```

Five V5 containers are currently running under Compose project `cyberwm_v5`; HTTP, SSH, source-specific block, and firewall cleanup smoke checks pass. The old V4 network was stopped because both use the authorized `10.77.0.0/24` range.

## Required next step

Switch Pi to `openai-codex/gpt-6-astra` now for one methodology/code review before capture. Astra should:

1. read `docs/V5_PLAN.md`, plan/split CSVs, validator, compose file, and both V5 shell scripts completely;
2. audit scope feasibility, paired design, schedule margin, leakage, split balance, process cleanup, and shell safety;
3. run static tests and inspect the live five-container lab;
4. make only necessary pre-capture corrections;
5. add dynamic five-host sequence/action builders and tests;
6. capture one benign and one blocked-action smoke episode only after review;
7. freeze hashes only after both smoke episodes process and validate.

Do not capture the 80-episode corpus yet. Do not retune V3/V4.
