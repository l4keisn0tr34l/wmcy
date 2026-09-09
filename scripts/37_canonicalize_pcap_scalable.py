#!/usr/bin/env python3
"""Disk-bounded, order-independent PCAP/PCAPNG to canonical observable events."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import shutil
import socket
import struct
import sys
import tempfile
from typing import BinaryIO, Iterator, NamedTuple

PACKET = struct.Struct("!q4sH4sHBI4Bq")
AGGREGATE = struct.Struct("!q4sH4sHBQQQQQQqq")
CANONICAL_COLUMNS = [
    "event_id", "dataset_id", "source_file", "timestamp", "timestamp_resolution_sec",
    "source_ip", "source_port", "destination_ip", "destination_port", "protocol",
    "total_fwd_packets", "total_backward_packets", "total_length_of_fwd_packets",
    "total_length_of_bwd_packets", "syn_flag_count", "ack_flag_count", "rst_flag_count",
    "fin_flag_count", "flow_duration",
]


class CapturedPacket(NamedTuple):
    timestamp_ns: int
    data: bytes
    original_length: int
    linktype: int


def read_exact(handle: BinaryIO, size: int) -> bytes:
    value = handle.read(size)
    if len(value) != size: raise ValueError(f"truncated capture: wanted {size}, found {len(value)}")
    return value


def classic_packets(handle: BinaryIO, magic: bytes) -> Iterator[CapturedPacket]:
    formats = {
        b"\xd4\xc3\xb2\xa1": ("<", 1_000_000), b"\xa1\xb2\xc3\xd4": (">", 1_000_000),
        b"\x4d\x3c\xb2\xa1": ("<", 1_000_000_000), b"\xa1\xb2\x3c\x4d": (">", 1_000_000_000),
    }
    endian, resolution = formats[magic]
    header = read_exact(handle, 20); _, _, _, _, _, linktype = struct.unpack(endian + "HHIIII", header)
    packet_header = struct.Struct(endian + "IIII")
    while True:
        raw = handle.read(16)
        if not raw: return
        if len(raw) != 16: raise ValueError("truncated classic PCAP packet header")
        seconds, fraction, captured, original = packet_header.unpack(raw)
        if fraction >= resolution: raise ValueError("invalid PCAP timestamp fraction")
        data = read_exact(handle, captured)
        timestamp_ns = seconds * 1_000_000_000 + fraction * (1_000_000_000 // resolution)
        yield CapturedPacket(timestamp_ns, data, original, linktype)


def pcapng_options(data: bytes, endian: str) -> dict[int, list[bytes]]:
    values: dict[int, list[bytes]] = {}; offset = 0
    while offset + 4 <= len(data):
        code, length = struct.unpack_from(endian + "HH", data, offset); offset += 4
        if code == 0: break
        if offset + length > len(data): raise ValueError("truncated PCAPNG option")
        values.setdefault(code, []).append(data[offset:offset + length])
        offset += (length + 3) & ~3
    return values


def pcapng_packets(handle: BinaryIO, first_type: bytes) -> Iterator[CapturedPacket]:
    raw_type = first_type; endian: str | None = None; interfaces: list[tuple[int, int, int, int]] = []
    while raw_type:
        raw_length = read_exact(handle, 4)
        if raw_type == b"\x0a\x0d\x0d\x0a":
            byte_order = read_exact(handle, 4)
            if byte_order == b"\x4d\x3c\x2b\x1a": endian = "<"
            elif byte_order == b"\x1a\x2b\x3c\x4d": endian = ">"
            else: raise ValueError("invalid PCAPNG byte-order magic")
            total = struct.unpack(endian + "I", raw_length)[0]
            if total < 28 or total % 4: raise ValueError(f"invalid PCAPNG section length {total}")
            remainder = read_exact(handle, total - 12)
            if struct.unpack(endian + "I", remainder[-4:])[0] != total:
                raise ValueError("PCAPNG section length trailer mismatch")
            interfaces = []
        else:
            if endian is None: raise ValueError("PCAPNG block precedes section header")
            total = struct.unpack(endian + "I", raw_length)[0]
            if total < 12 or total % 4: raise ValueError(f"invalid PCAPNG block length {total}")
            remainder = read_exact(handle, total - 8)
            if struct.unpack(endian + "I", remainder[-4:])[0] != total:
                raise ValueError("PCAPNG block length trailer mismatch")
            body = remainder[:-4]
            block_type = struct.unpack(endian + "I", raw_type)[0]
            if block_type == 1:  # Interface Description Block
                if len(body) < 8: raise ValueError("short PCAPNG interface block")
                linktype, _, _ = struct.unpack_from(endian + "HHI", body)
                options = pcapng_options(body[8:], endian)
                resolution_byte = options.get(9, [b"\x06"])[0][0]
                if resolution_byte & 0x80:
                    units_per_second = 2 ** (resolution_byte & 0x7f)
                else:
                    units_per_second = 10 ** resolution_byte
                offset_seconds = 0
                if 14 in options:
                    offset_seconds = struct.unpack(endian + "q", options[14][0])[0]
                interfaces.append((linktype, units_per_second, offset_seconds, len(interfaces)))
            elif block_type == 6:  # Enhanced Packet Block
                if len(body) < 20: raise ValueError("short PCAPNG enhanced packet block")
                interface_id, high, low, captured, original = struct.unpack_from(endian + "IIIII", body)
                if interface_id >= len(interfaces): raise ValueError("unknown PCAPNG interface")
                if 20 + captured > len(body): raise ValueError("truncated PCAPNG packet data")
                linktype, units, offset_seconds, _ = interfaces[interface_id]
                ticks = (high << 32) | low
                timestamp_ns = (ticks * 1_000_000_000) // units + offset_seconds * 1_000_000_000
                yield CapturedPacket(timestamp_ns, body[20:20 + captured], original, linktype)
            elif block_type == 2:  # Obsolete Packet Block
                if len(body) < 20: raise ValueError("short PCAPNG packet block")
                interface_id, _, high, low, captured, original = struct.unpack_from(endian + "HHIIII", body)
                if interface_id >= len(interfaces): raise ValueError("unknown PCAPNG interface")
                linktype, units, offset_seconds, _ = interfaces[interface_id]
                ticks = (high << 32) | low
                timestamp_ns = (ticks * 1_000_000_000) // units + offset_seconds * 1_000_000_000
                yield CapturedPacket(timestamp_ns, body[20:20 + captured], original, linktype)
            # Simple packets have no timestamp and are deliberately not fabricated.
        raw_type = handle.read(4)
        if raw_type and len(raw_type) != 4: raise ValueError("truncated capture block type")


def capture_packets(path: Path) -> Iterator[CapturedPacket]:
    with path.open("rb") as handle:
        magic = read_exact(handle, 4)
        if magic in {b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4",
                     b"\x4d\x3c\xb2\xa1", b"\xa1\xb2\x3c\x4d"}:
            yield from classic_packets(handle, magic)
        elif magic == b"\x0a\x0d\x0d\x0a":
            yield from pcapng_packets(handle, magic)
        else:
            raise ValueError(f"unsupported capture magic: {magic.hex()}")


def ipv4_summary(packet: CapturedPacket) -> tuple[bytes, int, bytes, int, int, int, int, int, int] | None:
    data = packet.data
    if packet.linktype != 1: raise ValueError(f"unsupported link type {packet.linktype}; Ethernet(1) required")
    if len(data) < 14: return None
    offset = 14; ethertype = struct.unpack_from("!H", data, 12)[0]
    while ethertype in {0x8100, 0x88A8, 0x9100}:
        if len(data) < offset + 4: return None
        ethertype = struct.unpack_from("!H", data, offset + 2)[0]; offset += 4
    if ethertype != 0x0800 or len(data) < offset + 20: return None
    version_ihl = data[offset]
    if version_ihl >> 4 != 4: return None
    ihl = (version_ihl & 0x0F) * 4
    if ihl < 20 or len(data) < offset + ihl: return None
    protocol = data[offset + 9]; src = data[offset + 12:offset + 16]; dst = data[offset + 16:offset + 20]
    fragment = struct.unpack_from("!H", data, offset + 6)[0] & 0x1FFF
    transport = offset + ihl; sport = dport = syn = ack = rst = fin = 0
    if fragment == 0 and protocol == 6 and len(data) >= transport + 14:
        sport, dport = struct.unpack_from("!HH", data, transport)
        flags = data[transport + 13]
        fin, syn, rst, ack = (int(bool(flags & bit)) for bit in (0x01, 0x02, 0x04, 0x10))
    elif fragment == 0 and protocol == 17 and len(data) >= transport + 4:
        sport, dport = struct.unpack_from("!HH", data, transport)
    return src, sport, dst, dport, protocol, packet.original_length, syn, ack, rst, fin


def aggregate_partition(packet_path: Path, aggregate_path: Path) -> int:
    aggregate: dict[tuple[int, bytes, int, bytes, int, int], list[int]] = {}
    with packet_path.open("rb") as handle:
        while raw := handle.read(PACKET.size):
            if len(raw) != PACKET.size: raise ValueError(f"truncated spool partition {packet_path}")
            bucket, src, sport, dst, dport, protocol, length, syn, ack, rst, fin, timestamp = PACKET.unpack(raw)
            key = (bucket, src, sport, dst, dport, protocol)
            record = aggregate.get(key)
            if record is None:
                aggregate[key] = [1, length, syn, ack, rst, fin, timestamp, timestamp]
            else:
                record[0] += 1; record[1] += length; record[2] += syn; record[3] += ack
                record[4] += rst; record[5] += fin
                record[6] = min(record[6], timestamp); record[7] = max(record[7], timestamp)
    rows = [(key, values) for key, values in aggregate.items()]
    rows.sort(key=lambda item: (item[1][6], *item[0]))
    with aggregate_path.open("wb") as output:
        for (bucket, src, sport, dst, dport, protocol), values in rows:
            output.write(AGGREGATE.pack(bucket, src, sport, dst, dport, protocol, *values))
    return len(rows)


def read_aggregate(handle: BinaryIO) -> tuple[int | bytes, ...] | None:
    raw = handle.read(AGGREGATE.size)
    if not raw: return None
    if len(raw) != AGGREGATE.size: raise ValueError("truncated aggregate chunk")
    return AGGREGATE.unpack(raw)


def sort_key(row: tuple[int | bytes, ...]) -> tuple[object, ...]:
    # first timestamp is field 12; remaining fields make ties deterministic.
    return row[12], row[0], row[1], row[2], row[3], row[4], row[5]


def iso_timestamp(timestamp_ns: int) -> str:
    seconds, nanos = divmod(timestamp_ns, 1_000_000_000)
    base = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{nanos:09d}+00:00"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def merge_aggregates(paths: list[Path], output_path: Path, dataset_id: str,
                     source_file: str, bucket_seconds: str) -> int:
    handles = [path.open("rb") for path in paths]; heap = []
    try:
        for index, handle in enumerate(handles):
            row = read_aggregate(handle)
            if row is not None: heapq.heappush(heap, (sort_key(row), index, row))
        opener = gzip.open if str(output_path).endswith(".gz") else open
        with opener(output_path, "wt", newline="", encoding="utf-8") as output:
            writer = csv.writer(output); writer.writerow(CANONICAL_COLUMNS); event_id = 0
            while heap:
                _, index, row = heapq.heappop(heap)
                (bucket, src, sport, dst, dport, protocol, packets, byte_count,
                 syn, ack, rst, fin, first, last) = row
                writer.writerow([
                    event_id, dataset_id, source_file, iso_timestamp(int(first)), bucket_seconds,
                    socket.inet_ntoa(src), sport, socket.inet_ntoa(dst), dport, protocol,
                    packets, 0, byte_count, 0, syn, ack, rst, fin, (last - first) / 1_000_000_000,
                ])
                event_id += 1
                following = read_aggregate(handles[index])
                if following is not None: heapq.heappush(heap, (sort_key(following), index, following))
        return event_id
    finally:
        for handle in handles: handle.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pcap"); ap.add_argument("--dataset-id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--bucket-seconds", default="1.0")
    ap.add_argument("--partitions", type=int, default=64); ap.add_argument("--temp-dir")
    ap.add_argument("--max-packets", type=int, help="bounded parser smoke test only")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(); source = Path(args.pcap); output = Path(args.out); manifest_path = Path(args.manifest)
    if not source.is_file(): raise FileNotFoundError(source)
    if output.resolve() == manifest_path.resolve(): raise ValueError("output and manifest paths must differ")
    if not args.force and (output.exists() or manifest_path.exists()):
        raise FileExistsError("refusing to replace output/manifest without --force")
    if args.partitions < 2: raise ValueError("partitions must be at least two")
    bucket_ns = int(Decimal(args.bucket_seconds) * Decimal(1_000_000_000))
    if bucket_ns <= 0: raise ValueError("bucket seconds must be positive")
    output.parent.mkdir(parents=True, exist_ok=True); manifest_path.parent.mkdir(parents=True, exist_ok=True)
    work_parent = Path(args.temp_dir) if args.temp_dir else output.parent
    work_parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pcap-canonical-", dir=work_parent))
    packet_paths = [work / f"packets-{index:04d}.bin" for index in range(args.partitions)]
    handles = [path.open("wb") for path in packet_paths]
    total = ipv4 = skipped = inversions = 0; maximum_backward_ns = 0; previous = None
    capture_min = capture_max = ipv4_min = ipv4_max = None
    try:
        for packet in capture_packets(source):
            if args.max_packets is not None and total >= args.max_packets: break
            total += 1
            capture_min = packet.timestamp_ns if capture_min is None else min(capture_min, packet.timestamp_ns)
            capture_max = packet.timestamp_ns if capture_max is None else max(capture_max, packet.timestamp_ns)
            if previous is not None and packet.timestamp_ns < previous:
                inversions += 1; maximum_backward_ns = max(maximum_backward_ns, previous - packet.timestamp_ns)
            previous = packet.timestamp_ns
            summary = ipv4_summary(packet)
            if summary is None: skipped += 1; continue
            src, sport, dst, dport, protocol, length, syn, ack, rst, fin = summary
            bucket = packet.timestamp_ns // bucket_ns; partition = bucket % args.partitions
            handles[partition].write(PACKET.pack(
                bucket, src, sport, dst, dport, protocol, length, syn, ack, rst, fin, packet.timestamp_ns
            )); ipv4 += 1
            ipv4_min = packet.timestamp_ns if ipv4_min is None else min(ipv4_min, packet.timestamp_ns)
            ipv4_max = packet.timestamp_ns if ipv4_max is None else max(ipv4_max, packet.timestamp_ns)
        for handle in handles: handle.close()
        aggregate_paths = []; partition_rows = []
        for index, packet_path in enumerate(packet_paths):
            aggregate_path = work / f"aggregate-{index:04d}.bin"
            rows = aggregate_partition(packet_path, aggregate_path)
            aggregate_paths.append(aggregate_path); partition_rows.append(rows)
            packet_path.unlink()
        temporary_output = work / ("observations.csv.gz" if str(output).endswith(".gz") else "observations.csv")
        rows = merge_aggregates(aggregate_paths, temporary_output, args.dataset_id, source.name, args.bucket_seconds)
        output_sha256 = file_sha256(temporary_output)
        report = {
            "dataset_id": args.dataset_id, "source_file": str(source.resolve()),
            "source_size_bytes": source.stat().st_size, "bucket_seconds": args.bucket_seconds,
            "partitions": args.partitions, "max_packets": args.max_packets,
            "capture_packets_read": total, "ipv4_packets": ipv4,
            "non_ipv4_or_malformed_packets_skipped": skipped,
            "capture_timestamp_min": iso_timestamp(capture_min) if capture_min is not None else None,
            "capture_timestamp_max": iso_timestamp(capture_max) if capture_max is not None else None,
            "ipv4_timestamp_min": iso_timestamp(ipv4_min) if ipv4_min is not None else None,
            "ipv4_timestamp_max": iso_timestamp(ipv4_max) if ipv4_max is not None else None,
            "timestamp_inversions": inversions,
            "maximum_adjacent_backward_seconds": maximum_backward_ns / 1_000_000_000,
            "canonical_rows": rows, "partition_aggregate_rows": partition_rows,
            "output_sha256": output_sha256,
            "observations_only": True, "labels_included": False,
            "parser_scope": "classic PCAP or PCAPNG; Ethernet IPv4; TCP/UDP ports and TCP SYN/ACK/RST/FIN",
            "out_of_order_policy": "bucket-modulo disk partitioning then sorted k-way merge",
        }
        temporary_manifest = work / "manifest.json"
        temporary_manifest.write_text(json.dumps(report, indent=2) + "\n")
        os.replace(temporary_output, output); os.replace(temporary_manifest, manifest_path)
        print(f"read {total:,} packets ({ipv4:,} IPv4); inversions={inversions:,}")
        print(f"wrote {rows:,} canonical events -> {output}")
        print(f"manifest -> {manifest_path}"); return 0
    finally:
        for handle in handles:
            if not handle.closed: handle.close()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__": sys.exit(main())
