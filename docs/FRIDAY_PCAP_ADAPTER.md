# Friday Working-Hours PCAP Adapter

## Status

Full observable-only canonicalization completed successfully.

Raw source (immutable, outside Git):

```text
/home/paprika/Documents/153/ds/Friday-WorkingHours.pcap
8,839,309,056 bytes
SHA-256 beff0dcce1eebc9b2454582f4dc8ed0ba0112b2c619a710bf03af93147254cd0
```

Despite its `.pcap` suffix, the file is PCAPNG.

## Why a new converter was required

`scripts/06_pcap_to_canonical.py` retains every bucket/5-tuple aggregate in one Python dictionary. That is transparent for small lab captures but not safely bounded for ten million packets. It also converts timestamps through float, which creates false sub-microsecond digits.

`scripts/37_canonicalize_pcap_scalable.py` instead:

1. streams classic PCAP or PCAPNG blocks;
2. parses Ethernet/VLAN + IPv4 + TCP/UDP primitives;
3. preserves integer micro/nanosecond timestamps;
4. writes fixed-width packet summaries to bucket-modulo disk partitions;
5. aggregates each bounded partition by one-second bucket + directed 5-tuple;
6. sorts each aggregate chunk;
7. k-way merges chunks into one chronological gzip CSV;
8. atomically installs output and manifest;
9. deletes temporary spool data even after failure.

Every packet in one bucket maps to the same partition, so packet order cannot split an aggregate.

## Full observed result

```text
capture packets:                   9,997,874
IPv4 packets aggregated:           9,915,680
non-IPv4/malformed skipped:           82,194
adjacent timestamp inversions:         7,094
maximum adjacent backward jump:     0.000014 s
canonical directed events:         2,102,560
capture timestamp range:           2017-07-07 11:59:39.599128 UTC
                                   2017-07-07 20:02:41.169108 UTC
IPv4 canonical timestamp range:    2017-07-07 11:59:50.315195 UTC
                                   2017-07-07 20:02:41.073211 UTC
elapsed conversion time:           1m20.72s
maximum RSS:                        47,480 KB
compressed canonical size:         ~32 MB
```

Output:

```text
outputs/cic2017_friday_pcap/canonical/observations.csv.gz
outputs/cic2017_friday_pcap/canonical/manifest.json
```

Output SHA-256:

```text
e099281ef8903ed0697a3e6612935f503e4f68cec926274a0dac6813f9fc64a5
```

A chunked integrity audit verifies:

- contiguous event IDs;
- chronological timestamps;
- finite numeric values;
- `sum(total_fwd_packets) = 9,915,680`;
- flag counts never exceed packet counts;
- durations are inside the one-second bucket;
- no label/ATT&CK/LM columns.

## Regression evidence

`scripts/38_test_scalable_pcap_adapter.py` compares native output with the Scapy adapter on `lab_001`:

- all 68 directed bucket/5-tuple keys match;
- packet, byte, SYN, ACK, RST, and FIN aggregates match exactly;
- timestamps/durations agree within 0.5 microseconds;
- native times remain exact microsecond multiples rather than float artifacts;
- a 1,000-packet PCAPNG smoke parse passes.

A separate 100,000-packet Friday conversion produced 14,359 canonical events in 1.35 seconds with about 17 MB RSS.

## Observable output only

The adapter never reads CIC labels. It emits packet-derived observable primitives only. Any Friday CSV labels must remain in a separate truth table and are not yet aligned to this output. No ATT&CK or LM truth is claimed.

## Limitations

- Ethernet link type only;
- IPv4 only; IPv6 and non-IP traffic are skipped;
- TCP/UDP ports plus TCP SYN/ACK/RST/FIN only;
- non-initial IP fragments use zero ports;
- `flow_duration` is first-to-last packet time **inside one bucket**, not whole connection duration;
- each event is directed, so backward counters remain zero;
- no application payload parsing;
- no host truth or exact progression labels.

The next step is a causal enterprise graph-state adapter. The current three-host roster contract cannot simply be reused on a capture with many IPs; context-only host selection/anonymization must be specified before sequence construction.
