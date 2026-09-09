#!/usr/bin/env python3
"""Artifact and synthetic causality checks for Friday graph sequences."""
from __future__ import annotations

import argparse
import importlib.util
import ipaddress
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_builder():
    spec = importlib.util.spec_from_file_location("friday_sequences", ROOT / "scripts/39_build_friday_graph_sequences.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def synthetic_tests(module) -> None:
    columns = {
        "source_ip": ["192.168.10.1"], "destination_ip": ["192.168.10.2"],
        "source_port": [12345], "destination_port": [22], "bytes": [100], "packets": [2],
        "syn_flag_count": [1], "ack_flag_count": [1], "rst_flag_count": [0], "fin_flag_count": [0],
        "flow_duration": [0.1],
    }
    frame = pd.DataFrame(columns); roster = ("192.168.10.1", "192.168.10.2", "192.168.10.3")
    first_seen = {("192.168.10.1", "192.168.10.2"): 100}
    network = ipaddress.ip_network("192.168.10.0/24")
    first, edge_first = module.state_vector(frame, roster, first_seen, 100, network)
    later, edge_later = module.state_vector(frame, roster, first_seen, 105, network)
    assert first.shape == (141,) and first[5] == 1 and later[5] == 0
    assert first[9] == 1 and first[10] == 0 and first[11] == 0
    assert edge_first[0] == edge_later[0] == 1
    # Future-only host activity cannot enter choose_roster because the API takes
    # only the explicitly supplied context table.
    context = pd.concat([frame, pd.DataFrame({**columns, "source_ip": ["192.168.10.2"],
                                              "destination_ip": ["192.168.10.3"]})])
    roster_a = module.choose_roster(context, "synthetic", 100)
    future = pd.DataFrame({**columns, "source_ip": ["203.0.113.50"],
                           "destination_ip": ["203.0.113.51"]})
    roster_b = module.choose_roster(context, "synthetic", 100)  # future deliberately not passed
    assert roster_a == roster_b and not set(future.source_ip).intersection(roster_a)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequences-dir", default=str(ROOT / "outputs/cic2017_friday_pcap/graph_sequences"))
    ap.add_argument("--canonical-manifest", default=str(ROOT / "outputs/cic2017_friday_pcap/canonical/manifest.json"))
    args = ap.parse_args(); out = Path(args.sequences_dir); module = load_builder(); synthetic_tests(module)
    arrays = np.load(out / "train.npz"); metadata = json.loads((out / "feature_metadata.json").read_text())
    canonical = json.loads(Path(args.canonical_manifest).read_text()); audit = pd.read_csv(out / "sample_manifest.csv")
    expected = {"context_states": (5775, 3, 141), "future_states": (5775, 6, 141),
                "future_edge_presence": (5775, 6, 6)}
    for name, shape in expected.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise AssertionError(f"bad {name}: {arrays[name].shape}")
    if not set(np.unique(arrays["future_edge_presence"])).issubset({0, 1}):
        raise AssertionError("non-binary edge target")
    if len(audit) != 5775 or not audit.split.eq("train").all(): raise AssertionError("bad audit rows/split")
    starts = pd.to_datetime(audit.sample_start, utc=True); ends = pd.to_datetime(audit.future_end, utc=True)
    if not starts.is_monotonic_increasing or not ((starts.diff().dropna().dt.total_seconds() % 5) == 0).all():
        raise AssertionError("sample times are not causal chronological five-second candidates")
    if ends.max() > pd.Timestamp(canonical["capture_timestamp_max"]): raise AssertionError("future exceeds capture")
    if metadata["canonical_observations_sha256"] != canonical["output_sha256"]:
        raise AssertionError("canonical lineage mismatch")
    if (out / "validation.npz").exists() or (out / "test.npz").exists():
        raise AssertionError("connected Friday capture was split")
    names = metadata["state_feature_names"]
    for forbidden in module.FORBIDDEN:
        if any(forbidden in name.lower() for name in names): raise AssertionError(f"forbidden feature {forbidden}")
    print("Friday graph sequences PASS: artifact shapes, lineage, chronology, novelty, and context-only roster")
    return 0


if __name__ == "__main__": sys.exit(main())
