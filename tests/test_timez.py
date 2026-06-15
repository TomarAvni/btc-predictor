"""Unit tests for Israel-time display helpers."""

from __future__ import annotations

import os
import sys
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.timez import (  # noqa: E402
    format_israel,
    now_israel_str,
    to_israel,
    utc_str_to_israel,
)


class TestTimez(unittest.TestCase):
    def test_summer_is_idt_utc_plus_3(self) -> None:
        # June -> Israel Daylight Time (UTC+3).
        dt = datetime(2026, 6, 15, 6, 0, tzinfo=timezone.utc)
        out = format_israel(dt)
        self.assertIn("09:00", out)
        self.assertIn("IDT", out)

    def test_winter_is_ist_utc_plus_2(self) -> None:
        # January -> Israel Standard Time (UTC+2).
        dt = datetime(2026, 1, 15, 6, 0, tzinfo=timezone.utc)
        out = format_israel(dt)
        self.assertIn("08:00", out)
        self.assertIn("IST", out)

    def test_naive_datetime_treated_as_utc(self) -> None:
        naive = datetime(2026, 6, 15, 6, 0)
        self.assertEqual(to_israel(naive).hour, 9)

    def test_parse_log_header_format(self) -> None:
        out = utc_str_to_israel("2026-06-15 06:00 UTC")
        self.assertIn("09:00", out)

    def test_parse_iso_format(self) -> None:
        out = utc_str_to_israel("2026-01-15T06:00:00+00:00")
        self.assertIn("08:00", out)

    def test_bad_input_returns_fallback(self) -> None:
        self.assertEqual(utc_str_to_israel(None), "—")
        self.assertEqual(utc_str_to_israel(""), "—")

    def test_unparseable_returns_original(self) -> None:
        self.assertEqual(utc_str_to_israel("not a date"), "not a date")

    def test_now_israel_str_has_label(self) -> None:
        out = now_israel_str()
        self.assertTrue(out.endswith("IDT") or out.endswith("IST") or "Israel" in out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
