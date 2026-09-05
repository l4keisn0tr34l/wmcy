#!/usr/bin/env python3
"""Create exact-timestamp canonical directed flow events from a lab PCAP.

This is intentionally a small, transparent flow builder for our controlled synthetic/emulated
lab. It aggregates packets by fixed time bucket + directed 5-tuple. It is NOT intended to
reproduce every CICFlowMeter feature; the permanent graph-state builder uses common primitive
features that are available in both CIC and this lab representation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap")
    ap.add_argument("--bucket-seconds", type=float, default=1.0)
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    agg = defaultdict(lambda: {
        "packets": 0, "bytes": 0,
        "syn": 0, "ack": 0, "rst": 0, "fin": 0,
        "first_ts": None, "last_ts": None,
    })

    with PcapReader(args.pcap) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue
            ts = float(pkt.time)
            bucket = int(ts // args.bucket_seconds) * args.bucket_seconds
            ip = pkt[IP]
            proto = int(ip.proto)
            sport = dport = 0
            syn = ack = rst = fin = 0
            if TCP in pkt:
                sport, dport = int(pkt[TCP].sport), int(pkt[TCP].dport)
                flags = int(pkt[TCP].flags)
                fin = int(bool(flags & 0x01))
                syn = int(bool(flags & 0x02))
                rst = int(bool(flags & 0x04))
                ack = int(bool(flags & 0x10))
            elif UDP in pkt:
                sport, dport = int(pkt[UDP].sport), int(pkt[UDP].dport)

            key = (bucket, ip.src, sport, ip.dst, dport, proto)
            rec = agg[key]
            rec["packets"] += 1
            rec["bytes"] += len(pkt)
            rec["syn"] += syn; rec["ack"] += ack; rec["rst"] += rst; rec["fin"] += fin
            rec["first_ts"] = ts if rec["first_ts"] is None else min(rec["first_ts"], ts)
            rec["last_ts"] = ts if rec["last_ts"] is None else max(rec["last_ts"], ts)

    rows = []
    for event_id, (key, rec) in enumerate(sorted(agg.items(), key=lambda kv: kv[0])):
        bucket, src, sport, dst, dport, proto = key
        rows.append({
            "event_id": event_id,
            "dataset_id": args.dataset_id,
            "source_file": Path(args.pcap).name,
            "timestamp": pd.to_datetime(rec["first_ts"], unit="s", utc=True),
            "timestamp_resolution_sec": args.bucket_seconds,
            "source_ip": src,
            "source_port": sport,
            "destination_ip": dst,
            "destination_port": dport,
            "protocol": proto,
            "total_fwd_packets": rec["packets"],
            "total_backward_packets": 0,
            "total_length_of_fwd_packets": rec["bytes"],
            "total_length_of_bwd_packets": 0,
            "syn_flag_count": rec["syn"],
            "ack_flag_count": rec["ack"],
            "rst_flag_count": rec["rst"],
            "fin_flag_count": rec["fin"],
            "flow_duration": max(0.0, rec["last_ts"] - rec["first_ts"]),
        })

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, compression="gzip" if str(out).endswith(".gz") else None)
    print(f"wrote {len(rows):,} exact-time canonical flow events -> {out}")


if __name__ == "__main__":
    main()
