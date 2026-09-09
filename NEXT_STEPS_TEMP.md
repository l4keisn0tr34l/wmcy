# Temporary Next-Steps Handoff

> Overwrite this file before each new code-writing batch.

## V4 milestone complete

Action and passive evaluations are frozen and inspected. Do not retrain, retune thresholds, or rerun either sealed evaluator.

Key action result: on five eligible permit/block pairs, scratch state MAE is `.0603`, edge/LM/pair AP is `1.0`, and the factual action has lower state error than the opposite action on `10/10` contexts. Friday initialization has worse state MAE `.0719`.

Key passive result: across 12 passive/direct episodes, the frozen branch model alerts on `3/3` stopped prefixes but `0/3` progressing prefixes before a positive horizon and `0/3` direct-credential progressions. Window LM F1/AP is `.483/.472`.

On the same 10 action contexts, passive/action LM F1 is directly comparable at `.400/1.000`. Passive/action normalized state MAEs `.141/.060` use different V3/V4 training scalers and must not be presented as a direct improvement ratio. The within-action-model factual-versus-opposite comparison is valid.

## Report

```text
/home/paprika/Downloads/rssm_eod_report.html
outputs/mvp_v2/report/rssm_eod_report.html
SHA-256 7b886a2d0da7bbc856deda7a621f33d9ff34eda22197bcc79698ba1597fe25e5
```

## Next work

1. Do not tune against V3 or V4.
2. Design a new corpus with additional topologies, host counts, background processes, direct-credential paths, and interventions.
3. Create topology/scenario-disjoint train/validation/test before model changes.
4. Obtain additional precise raw PCAP/Zeek captures with disconnected capture groups; avoid label-only shuffled IDS CSVs.
5. Evaluate public initialization across multiple captures.
6. Add calibration only after enough independent validation episodes exist.
