"""Parse defender-observable OpenSSH container logs into canonical V6 auth events."""
from __future__ import annotations

from datetime import datetime
import re
from typing import Iterable

from src.cyberwm.v6_contract import HOSTS

AUTH = re.compile(
    r"^(?P<result>Accepted|Failed) (?P<method>password|publickey) for "
    r"(?:(?:invalid user) )?\S+ from (?P<remote>\d{1,3}(?:\.\d{1,3}){3}) "
    r"port (?P<port>\d+) ssh2(?:[: ].*)?$"
)


def utc_timestamp(value: str) -> datetime:
    if not (value.endswith("Z") or value.endswith("+00:00")):
        raise ValueError(f"authentication timestamp must be literal UTC: {value}")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else ""))
    except ValueError as error:
        raise ValueError(f"invalid authentication timestamp: {value}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"authentication timestamp must be explicit UTC: {value}")
    return parsed


def canonical_utc(value: str) -> str:
    """Retain source fractional precision; only normalize Docker's Z suffix."""
    return value.removesuffix("Z") + ("+00:00" if value.endswith("Z") else "")


def parse_auth_logs(host: str, lines: Iterable[str], capture_start: str, capture_end: str
                    ) -> list[dict[str, str | int]]:
    """Parse timestamped `docker logs --timestamps` output.

    Usernames/fingerprints/raw messages are deliberately discarded. Non-auth SSH
    lifecycle messages are ignored rather than converted to invented events.
    """
    if host not in HOSTS:
        raise ValueError(f"host outside V6 inventory: {host}")
    start = utc_timestamp(capture_start); end = utc_timestamp(capture_end)
    if end <= start:
        raise ValueError("capture end must follow start")
    inventory_by_ip = {ip: name for name, ip in HOSTS.items()}
    events: list[dict[str, str | int]] = []
    seen: set[tuple[datetime, str, str, str, int]] = set()
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"line {line_number}: missing Docker UTC timestamp")
        try:
            timestamp = utc_timestamp(parts[0])
        except ValueError as error:
            raise ValueError(f"line {line_number}: {error}") from error
        match = AUTH.match(parts[1])
        if not match:
            continue
        if timestamp < start or timestamp > end:
            raise ValueError(f"line {line_number}: auth event outside capture bounds")
        remote_ip = match.group("remote")
        if remote_ip not in inventory_by_ip:
            raise ValueError(f"line {line_number}: auth source outside isolated inventory: {remote_ip}")
        port = int(match.group("port"))
        key = (timestamp, match.group("result"), match.group("method"), remote_ip, port)
        if key in seen:
            raise ValueError(f"line {line_number}: duplicate auth event")
        seen.add(key)
        events.append({"event_time": canonical_utc(parts[0]),
            "host": host, "host_ip": HOSTS[host], "remote_host": inventory_by_ip[remote_ip],
            "remote_ip": remote_ip,
            "event_type": "auth_success" if match.group("result") == "Accepted" else "auth_failure",
            "auth_method": match.group("method"), "service": "ssh", "remote_port": port,
            "source": "openssh_docker_log"})
    events.sort(key=lambda row: (str(row["event_time"]), str(row["host"]), int(row["remote_port"])))
    return events


def validate_auth_event_schema(events: list[dict[str, str | int]]) -> None:
    required = {"event_time", "host", "host_ip", "remote_host", "remote_ip", "event_type",
                "auth_method", "service", "remote_port", "source"}
    for index, event in enumerate(events):
        if set(event) != required:
            raise ValueError(f"auth event {index}: schema differs")
        if event["host"] not in HOSTS or event["remote_host"] not in HOSTS:
            raise ValueError(f"auth event {index}: inventory reference differs")
        if event["event_type"] not in {"auth_success", "auth_failure"}:
            raise ValueError(f"auth event {index}: event type differs")
        if event["auth_method"] not in {"password", "publickey"} or event["service"] != "ssh":
            raise ValueError(f"auth event {index}: method/service differs")
        utc_timestamp(str(event["event_time"]))
        forbidden = {"scenario", "seed", "cohort", "split", "action", "ground_truth", "technique", "label"}
        if set(event) & forbidden:
            raise ValueError(f"auth event {index}: truth/audit leakage")
