#!/usr/bin/env python3
"""Regression checks for the disk-bounded native capture adapter."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_adapter():
    spec = importlib.util.spec_from_file_location("pcap_adapter", ROOT / "scripts/37_canonicalize_pcap_scalable.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classic-pcap", default=str(ROOT / "lab/episodes/lab_001/network.pcap"))
    ap.add_argument("--pcapng", default="/home/paprika/Documents/153/ds/Friday-WorkingHours.pcap")
    args = ap.parse_args(); classic = Path(args.classic_pcap); pcapng = Path(args.pcapng)
    if not classic.is_file(): raise FileNotFoundError(classic)
    with tempfile.TemporaryDirectory(prefix="pcap-adapter-test-") as temporary:
        work = Path(temporary); old = work / "old.csv.gz"; new = work / "new.csv.gz"
        subprocess.run([
            sys.executable, str(ROOT / "scripts/06_pcap_to_canonical.py"), str(classic),
            "--bucket-seconds", "1", "--dataset-id", "adapter_test", "--out", str(old),
        ], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([
            sys.executable, str(ROOT / "scripts/37_canonicalize_pcap_scalable.py"), str(classic),
            "--bucket-seconds", "1", "--dataset-id", "adapter_test", "--partitions", "8",
            "--out", str(new), "--manifest", str(work / "manifest.json"),
        ], check=True, stdout=subprocess.DEVNULL)
        reference = pd.read_csv(old); candidate = pd.read_csv(new)
        for frame in [reference, candidate]:
            timestamp = pd.to_datetime(frame.timestamp, utc=True)
            frame["bucket"] = timestamp.astype("int64") // 1_000_000_000
        keys = ["bucket", "source_ip", "source_port", "destination_ip", "destination_port", "protocol"]
        joined = reference.merge(candidate, on=keys, suffixes=("_reference", "_native"), validate="one_to_one")
        if len(joined) != len(reference) or len(joined) != len(candidate): raise AssertionError("flow key mismatch")
        exact = ["total_fwd_packets", "total_length_of_fwd_packets", "syn_flag_count",
                 "ack_flag_count", "rst_flag_count", "fin_flag_count"]
        for name in exact:
            if not joined[f"{name}_reference"].equals(joined[f"{name}_native"]):
                raise AssertionError(f"aggregate mismatch: {name}")
        reference_ns = pd.to_datetime(joined.timestamp_reference, utc=True).astype("int64")
        native_ns = pd.to_datetime(joined.timestamp_native, utc=True).astype("int64")
        # The old Scapy converter goes through float and fabricates sub-microsecond
        # digits. Native classic-PCAP values preserve the actual microsecond ticks.
        if not (native_ns % 1_000 == 0).all(): raise AssertionError("microsecond PCAP gained fake precision")
        if int((reference_ns - native_ns).abs().max()) > 500: raise AssertionError("timestamp differs by >0.5us")
        duration_delta = np.abs(joined.flow_duration_reference - joined.flow_duration_native)
        if float(duration_delta.max()) > 5e-7: raise AssertionError("duration differs by >0.5us")
    adapter = load_adapter()
    if pcapng.is_file():
        packets = []
        for index, packet in enumerate(adapter.capture_packets(pcapng)):
            packets.append(packet)
            if index == 999: break
        if len(packets) != 1_000 or not all(packet.linktype == 1 for packet in packets):
            raise AssertionError("PCAPNG/Ethernet smoke failed")
        if min(packet.timestamp_ns for packet in packets) <= 0: raise AssertionError("invalid PCAPNG timestamp")
    print(f"scalable PCAP adapter PASS: {len(candidate)} classic canonical rows; PCAPNG smoke={pcapng.is_file()}")
    return 0


if __name__ == "__main__": sys.exit(main())
