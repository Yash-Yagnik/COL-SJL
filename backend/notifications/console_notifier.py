"""
Simple notification module for the MVP.

When an intrusion-like event is detected, this module prints a
human-readable alert to the console. In a real deployment you can
extend this to send SMS, email, or push notifications.
"""

from __future__ import annotations

from typing import Dict, List

from backend.reporting.report_generator import IncidentReportGenerator


class ConsoleNotifier:
    """
    Console-based notifier used for development and early testing.
    """

    def __init__(self) -> None:
        self._report_generator = IncidentReportGenerator()

    def notify_if_intrusion(self, high_level_events: List[Dict]) -> None:
        """
        Inspect high-level events and, if a possible intrusion is
        present, print an alert to the console.
        """
        # Generate a structured report from the events.
        structured_report = self._report_generator.generate_structured_report(
            high_level_events
        )
        if not structured_report:
            return

        # Treat any report whose incident_type contains "Intrusion"
        # as an intrusion for this MVP.
        if "Intrusion" not in structured_report["incident_type"]:
            return

        text = self._report_generator.format_text_report(structured_report)

        print("=== SECURITY ALERT ===")
        print(text)
        print("======================")

