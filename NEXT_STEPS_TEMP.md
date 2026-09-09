# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## Checkpoint

Clean/pushed at `1f39445`. V4 capture is ready but blocked on interactive Docker rebuild, laptop uptime, and foreground sudo. No V4 episodes exist yet.

## Current noninteractive batch: scalable Friday PCAP canonicalizer

Implement `scripts/37_canonicalize_pcap_scalable.py` for the 8.3GB/9,997,874-packet Friday capture:

1. Read classic PCAP records with a bounded native Ethernet/IPv4/TCP/UDP parser; reject unsupported link types/formats explicitly.
2. Preserve exact micro/nanosecond packet times and count out-of-order timestamp inversions.
3. Spool fixed-width packet summaries into bucket-modulo disk partitions so all packets for one time bucket share a partition regardless of input order.
4. Aggregate each partition independently by bucket + directed 5-tuple.
5. Sort each bounded aggregate chunk and k-way merge chunks into one chronological gzip canonical file with stable event IDs.
6. Write an audit manifest with packet/IPv4 counts, inversion count/max backward jump, row count, parser limitations, and command settings.
7. Use atomic output installation and always clean temporary spool files.
8. Keep CIC labels entirely separate; this converter produces observables only.
9. Add `--max-packets` for bounded smoke tests.
10. Validate native output against `scripts/06_pcap_to_canonical.py` on small lab captures, including counts, keys, flags, bytes, timestamps, and durations.
11. Run only a bounded Friday smoke conversion now. Do not launch the full 8.3GB job without a fresh disk/time check.
