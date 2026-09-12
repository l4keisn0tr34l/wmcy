#!/usr/bin/env python3
"""Synthetic-only tests for V6 OpenSSH observable parsing."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cyberwm.v6_auth import parse_auth_logs, validate_auth_event_schema

START = "2026-10-01T12:00:00.000000+00:00"
END = "2026-10-01T12:02:30.000000+00:00"


class V6AuthTests(unittest.TestCase):
    def test_parse_success_failure_methods_and_drop_sensitive_text(self):
        lines = [
            "2026-10-01T12:00:05.123456789Z Server listening on 0.0.0.0 port 22.",
            "2026-10-01T12:00:40.100000000Z Failed password for lab from 10.77.0.20 port 42001 ssh2",
            "2026-10-01T12:01:20.200000000Z Accepted publickey for secret-user from 10.77.0.60 port 42002 ssh2: ED25519 SHA256:secret",
            "2026-10-01T12:01:20.300000000Z pam_unix(sshd:session): session opened for user secret-user(uid=1000)",
        ]
        events = parse_auth_logs("srv1", lines, START, END)
        validate_auth_event_schema(events)
        self.assertEqual(len(events), 2)
        self.assertEqual([event["event_type"] for event in events], ["auth_failure", "auth_success"])
        self.assertEqual([event["auth_method"] for event in events], ["password", "publickey"])
        self.assertEqual(events[0]["remote_host"], "ws1")
        self.assertEqual(events[1]["remote_host"], "jump1")
        self.assertNotIn("secret-user", str(events)); self.assertNotIn("SHA256", str(events))
        self.assertNotIn("session", str(events))

    def test_invalid_user_failure_is_observable_but_username_discarded(self):
        events = parse_auth_logs("srv2", [
            "2026-10-01T12:00:41.000000000Z Failed password for invalid user nobody from 10.77.0.25 port 99 ssh2"
        ], START, END)
        self.assertEqual(events[0]["event_type"], "auth_failure")
        self.assertNotIn("nobody", str(events))

    def test_reject_external_source(self):
        with self.assertRaisesRegex(ValueError, "outside isolated inventory"):
            parse_auth_logs("srv1", [
                "2026-10-01T12:00:41.000000000Z Failed password for lab from 8.8.8.8 port 99 ssh2"
            ], START, END)

    def test_reject_out_of_bounds_event(self):
        with self.assertRaisesRegex(ValueError, "outside capture bounds"):
            parse_auth_logs("srv1", [
                "2026-10-01T12:03:00.000000000Z Accepted password for lab from 10.77.0.20 port 99 ssh2"
            ], START, END)

    def test_reject_missing_or_non_utc_timestamp(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            parse_auth_logs("srv1", ["Accepted password for lab from 10.77.0.20 port 99 ssh2"], START, END)
        with self.assertRaisesRegex(ValueError, "explicit UTC"):
            parse_auth_logs("srv1", [
                "2026-10-01T12:00:40 Accepted password for lab from 10.77.0.20 port 99 ssh2"
            ], START, END)

    def test_reject_duplicate_and_unknown_host(self):
        line = "2026-10-01T12:00:40.000000000Z Accepted password for lab from 10.77.0.20 port 99 ssh2"
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_auth_logs("srv1", [line, line], START, END)
        with self.assertRaisesRegex(ValueError, "outside V6 inventory"):
            parse_auth_logs("external", [line], START, END)


if __name__ == "__main__":
    unittest.main()
