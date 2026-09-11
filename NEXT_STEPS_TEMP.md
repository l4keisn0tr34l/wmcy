# V5 smoke interruption and terminal repair

## Observed facts

- Tode shortcut setup at 2026-09-11 23:09:56 created unsupported Kitty action `copy_or_noop`; corrected user config to valid `copy_to_clipboard`, backed up original, and sent Kitty SIGUSR1 reload. Not a project/Astra/Docker change.
- v5_smoke_001 started at ~23:38:59 and was interrupted by a clean requested system reboot at ~23:40:52, before the 150s deadline. Journal: systemd-logind "The system will reboot now". AC online/charging; no OOM, panic, battery or thermal evidence. Request origin not established—do not speculate.
- Partial raw smoke had PCAP/background/truth headers and zero tcpdump drops but no completion metadata/cleanup receipt. Preserved as `_quarantine_v5_smoke_001_reboot_20260911T234648`; never process/resume as valid.
- Existing external inhibitor covered `sleep:idle`, not shutdown. Self-enforced `shutdown:sleep:idle` block inhibitor has now been added to smoke and corpus entry points; instructions/provenance/docs updated. Test and commit before recapture. No valid smoke, freeze, or corpus episode exists.

## Guardrails

No V3/V4 rerun/tuning, no V5 test access, no full corpus. Any code change makes prior smoke provenance obsolete; current smoke is already invalid. Overwrite neither raw capture nor quarantine. Re-run exactly two smoke IDs only after hardening. Interactive sudo remains user boundary.
