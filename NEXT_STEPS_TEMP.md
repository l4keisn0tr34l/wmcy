# V5 capture frozen after two validated real smokes

## Observed milestone

- v5_smoke_001 legitimate SSH: 150.000136s,29 dense states,184 observations,0 truth events/LM, five-host sustained mixed background, zero drops.
- v5_smoke_002 blocked scan/guess:150.000197s,29 dense states,228 observations; T1046 + T1110.001 + attempted T1021.004; chosen block before cutoff, applied after; PCAP actor SYN/pivot RST; zero completed LM; zero drops.
- Both raw/derived/general/V5 validators pass. Only five inventory IPs occur. Current source/image provenance matches. Containers stopped/firewall clean.
- Exports pass: passive[21,3,345], action[1,3,345], passive_action[1,3,345]; six-state futures;20 pairs. Action/passive_action common arrays exactly equal.
- Capture-only freeze created: configs/mvp_v5_capture_freeze.json SHA53069f98ed2f8e686f973da6aa6f62f22f4d9b735def1da9868e7247783a20df. 36 smoke file hashes, randomized80-ID capture order. Model protocol explicitly NOT FROZEN.
- First partial v5_smoke_001 attempt was user-confirmed manual shutdown, quarantined as `_quarantine_v5_smoke_001_reboot_20260911T234648`; never processed/trained.
- Kitty copy issue repaired outside repo: Tode generated unsupported copy_or_noop at ~/.config/kitty/tode/keybinds.kitty.conf; backed up and changed to copy_to_clipboard, Kitty reloaded.

## Invariants now frozen

Do not edit any path in src/cyberwm/v5_integrity.py FILES or rebuild Docker images during corpus. Freeze check must pass before every episode. Raw captures immutable; interrupted directory quarantine/restart from zero. Shutdown/sleep/idle inhibitor self-enforced. V3/V4 tests remain untouched. No V5 test export/model access.

## Immediate next action

Full corpus may now begin only when user is ready for ~3h20 raw capture + processing, laptop on AC/awake:

```bash
cd /home/paprika/Documents/153/wm
bash lab/generate_v5_corpus.sh
```

Interactive sudo required. Fixed hash order begins lab163, lab179, lab151... Pause ONLY between episodes with `touch lab/.pause_v5_corpus`; remove before resuming. Do not Ctrl+Z or shutdown during an active capture. On failure preserve raw directory and inspect/quarantine.

After80 episodes: inspect all raw/derived outputs and action semantics; build train/validation only. Then define and freeze common V5 train scaler, scratch/Friday/V4 initialization treatment, budgets, passive branch comparison, thresholds and metrics before any test export. Next critical model/evaluation protocol review merits Astra; Sol can execute routine capture/processing and verification.
