# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Current live state

- V2 generation is running.
- `lab_025` and `lab_026` are complete and derived/validated.
- `lab_027` was actively capturing when checked.
- Existing resume is episode-granular: completed episodes skip; interrupted mid-capture episodes require quarantine/regeneration.

## Current code batch

Add a graceful pause-at-boundary sentinel to `lab/generate_mvp_corpus.sh`:

```bash
touch lab/.pause_corpus
```

The runner should finish/process the current episode, detect the sentinel before starting the next episode, remove/acknowledge it, and exit successfully. Add the sentinel to `.gitignore` and document pause/resume commands. This change is for future invocations; do not assume the already-running shell reloads modified code.

## Required checks

- shell syntax;
- existing capture process remains untouched;
- pause check occurs only between episodes;
- no active PCAP is intentionally suspended;
- resume still validates/skips completed episodes;
- no external targets or sudo scope changes.

## Manual recovery for the current pre-feature invocation

If stopped with Ctrl+C during an episode, verify tcpdump stopped. If the active episode lacks `episode_metadata.csv`, preserve it under a suffixed ignored directory, then rerun the same plan. Completed episodes will skip.

## After V2 completes

Validate all 24 captures, build V2 manifests/sequences in separate outputs, rerun shortcut baselines, and then implement the compact passive RSSM smoke test. Senior implementation remains unavailable/unverified.
