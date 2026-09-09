# MITRE ATT&CK Mapping — Exact Role in This Project

## The central rule

MITRE mapping has two different meanings in this project:

1. **Creating ground truth** for known attack actions.
2. **Predicting future ATT&CK behavior** from learned network dynamics.

Do not confuse them.

---

# 1. How ground-truth MITRE mapping works

## Controlled lab data

We intentionally execute an action.

Example:

```text
ws1 runs network service discovery
```

The scenario is written to represent:

```text
T1046
Network Service Discovery
Discovery
```

The runner records that in:

```text
ground_truth.csv
```

The model is not involved.

Likewise:

```text
repeated wrong SSH passwords
```

is recorded as:

```text
T1110.001
Password Guessing
Credential Access
```

And successful SSH used to operate another machine is recorded as:

```text
T1021.004
SSH
Lateral Movement
```

This creates clean labels because the experimenter knows what action was executed.

---

# 2. Current lab ground truth

The parameterized lab generator can intentionally record:

```text
T1046      Network Service Discovery    Discovery
T1110.001  Password Guessing            Credential Access
T1021.004  SSH                           Lateral Movement
```

with randomized actor/pivot/target roles and exact controller timestamps. Benign ping and legitimate administrative SSH scenarios intentionally have no ATT&CK action truth. This provides necessary negative examples rather than teaching that every SSH connection is lateral movement.

V4 additionally distinguishes an intentionally executed but firewall-blocked `T1021.004` SSH attempt from completed lateral movement. The event keeps technique ID `T1021.004` but tactic text `Lateral Movement Attempt`; `has_lateral_movement` remains false because no session completes. This preserves attempted behavior without fabricating compromise success.

The current valid smoke-test `lab_003` records:

```text
srv1 -> internal subnet : T1046
srv1 -> srv2            : T1110.001
srv1 -> srv2            : T1021.004
srv2 -> ws1             : T1021.004
```

---

# 3. Public-dataset mapping

Public dataset labels are not automatically ATT&CK labels.

Process:

1. Read official scenario documentation.
2. Identify the actual action/behavior performed.
3. Identify actor, target, and time range if available.
4. Match the behavior to an ATT&CK technique definition.
5. Record:
   - technique ID;
   - technique name;
   - executed tactic/context;
   - confidence;
   - evidence/source.
6. If evidence is insufficient, use `UNKNOWN` or lower confidence.

Never do:

```text
Infiltration -> one guessed ATT&CK technique
```

just because the dataset has the word `Infiltration`.

---

# 4. Current CIC mapping file

Location:

```text
configs/cic2017_known_mitre.csv
```

Current high-confidence entries:

```text
FTP-Patator
 -> T1110.001 Password Guessing
 -> Credential Access
```

```text
SSH-Patator
 -> T1110.001 Password Guessing
 -> Credential Access
```

```text
Friday PortScan
 -> T1046 Network Service Discovery
 -> Discovery
```

The file also stores confidence/evidence.

---

# 5. Why technique and tactic are separate fields

ATT&CK techniques are the "how".

Tactics are the attacker objective/context.

A technique can be associated with more than one tactic.

Therefore do not hard-code a universal:

```text
technique -> exactly one tactic
```

for every possible action.

For controlled emulation, the executed scenario supplies the tactic/context in which the technique is used.

---

# 6. ATT&CK is not a rigid kill chain

Do not assume:

```text
Recon -> Initial Access -> PrivEsc -> Lateral -> C2 -> Exfil
```

must always happen in that exact order.

Real attacks can:

- skip stages;
- repeat discovery;
- branch to multiple hosts;
- perform tactics simultaneously;
- return to earlier objectives.

Therefore future ATT&CK output should usually be multi-label / probabilistic.

---

# 7. Temporal alignment

`05_align_ground_truth.py` does not "recognize" ATT&CK from packets.

It asks:

> Which known ATT&CK actions were active during this state window?

Conceptually:

```text
network states:
13:35:15-13:35:20
13:35:20-13:35:25
...

ground truth:
13:35:15-13:35:24 T1110.001

alignment:
both overlapping states receive T1110.001 truth
```

This gives state-level supervision without contaminating observation features.

---

# 8. What the model will eventually predict

Suppose the future horizon contains `K` ATT&CK techniques.

The model may output:

\[
\hat m_{t+h}\in[0,1]^K
\]

Example:

```text
Future +30 s

T1046      0.18
T1110.001  0.41
T1021.004  0.72
```

This is likely a multi-label head rather than a single softmax stage.

---

# 9. Why the MITRE head is not enough

This alone:

\[
S_{history}\rightarrow T1021.004
\]

is attack forecasting, but it is not a full world model.

The intended system also learns:

\[
S_{history}\rightarrow \hat S_{future}
\]

and:

\[
S_{history}\rightarrow \hat E_{future}
\]

where `E` represents graph edges.

MITRE then gives security meaning to the forecast.

---

# 10. Desired explainable coupling

A strong final prediction should be internally consistent.

Example:

```text
Predicted future network:
- novel internal edge wsA -> srvB
- predicted port/service pattern consistent with SSH
- source host's fan-out and credential-pressure history increased

Security interpretation:
- T1021.004 SSH / Lateral Movement probability = 0.72
```

This is better than a black-box label with no predicted network future.

---

# 11. Future ATT&CK data engineering

Planned:

1. Download/version a specific Enterprise ATT&CK STIX release.
2. Store it locally under a versioned `data/mitre/` or similar path.
3. Build a lookup utility:
   - technique ID;
   - name;
   - tactic relationships;
   - deprecated/revoked status.
4. Validate ground-truth scenario entries automatically.
5. Preserve the version used for every experiment.
6. Never silently remap old experiment labels when ATT&CK changes.

---

# 12. CALDERA future plan

A future emulation scale-up may use MITRE CALDERA.

Benefits:

- ATT&CK-aligned abilities;
- automated multi-stage operations;
- action timestamps;
- host/action metadata;
- reproducible playbook variants.

CALDERA ground truth should still be stored separately from defender-visible telemetry.

---

# 13. Mapping confidence policy

Suggested fields:

```text
technique_id
technique
tactic
confidence
evidence
mapping_source
attack_version
```

Confidence examples:

### High
Action is explicitly defined by the scenario/tool and maps directly to an ATT&CK technique.

### Medium
Public dataset documents behavior clearly but does not explicitly provide ATT&CK ID.

### Low
Behavior is inferred from incomplete narrative/label.

### Unknown
Evidence is insufficient.

Do not train high-confidence semantic heads on guessed truth without marking/handling the uncertainty.
