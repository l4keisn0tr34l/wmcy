# Data Sources — Role and Limitations

## Controlled Docker lab

### Role
Primary exact progression/lateral-movement truth.

### Raw
```text
lab/episodes/<id>/network.pcap
```

### Truth
```text
lab/episodes/<id>/ground_truth.csv
```

### Strength
Exact action/timing/source/target/ATT&CK context.

### Weakness
Currently tiny, scripted, and easy to memorize. Must scale/diversify.

---

## CICIDS2017

### Current inspected files
- Tuesday Working Hours.
- Thursday Afternoon Infiltration.
- Friday Afternoon PortScan.

### Observed copy properties
- 85 columns.
- minute-resolution timestamp export.
- flow-level labels.
- source/destination identities retained.

### Good uses
- flow/network dynamics;
- background traffic;
- known brute-force and scanning behavior;
- public-data pretraining;
- domain transfer.

### Bad use
Do not claim it supplies a clean second-by-second multi-stage lateral-movement trajectory.

### Friday precise-PCAP adapter

The local 8.839 GB Friday PCAPNG capture is fully canonicalized with exact packet timestamps. `scripts/37_canonicalize_pcap_scalable.py` processes 9,997,874 packets into 2,102,560 one-second directed events in a disk-bounded pipeline. It observed 7,094 adjacent timestamp inversions (maximum 14 microseconds), aggregates 9,915,680 IPv4 packets, and includes no labels. See `docs/FRIDAY_PCAP_ADAPTER.md`.

### Mapping truth
Use official scenario behavior and documented times, not label names alone.

---

## CSE-CIC-IDS2018

Verified local path:

```text
/home/paprika/Documents/153/ds/cicids2018/Processed Traffic Data for ML Algorithms
```

Ten large CICFlowMeter CSVs are present (approximately 6.5 GB total). A representative file has 80 fields and timestamps such as `14/02/2018 08:31:01`, but its header begins with destination port/protocol and does not expose source/destination IP fields. Therefore this processed form is not yet proven suitable for host graph construction.

### Intended role
- additional public flow dynamics;
- different attack/background distribution;
- pretraining and external evaluation.

Do not launch full preprocessing or claim graph suitability until all required identity/time fields are verified.

---

## UNSW-NB15

Verified local path:

```text
/home/paprika/Documents/153/ds/un/CSV Files
```

The four original `UNSW-NB15_*.csv` files total approximately 587 MB and contain 49 fields per row. Representative rows preserve source/destination IPs and ports, protocol, service, connection state, duration, bytes, and packet statistics. `NUSW-NB15_features.csv` provides field definitions. Reduced training/testing tables with 45 columns are also present.

### Verified temporal adapter

`scripts/26_canonicalize_unsw.py` has processed all 2,540,047 local original rows into five sorted, gap-bounded segments under two connected capture groups. Raw files contain 52k–108k timestamp inversions each, so sorting is essential. Source-file intervals overlap and must remain grouped to prevent split leakage. See `docs/UNSW_ADAPTER.md`.

### Intended role
- cross-dataset generalization;
- dynamics/background diversity;
- domain-shift test.

Raw categories remain separate truth and are not treated as exact ATT&CK/LM progression.

---

## MITRE ATT&CK

Knowledge base, not telemetry.

### Role
- technique/tactic vocabulary;
- mapping validation;
- semantic labels;
- eventual machine-readable lookup through Enterprise ATT&CK STIX.

It does not by itself give empirical network transition probabilities.

---

## Future CALDERA

Adversary-emulation platform.

### Intended role
Generate many ATT&CK-aligned episodes with controlled variation and precise action metadata.

### Important
CALDERA action metadata remains ground truth, separate from defender-visible network telemetry.
