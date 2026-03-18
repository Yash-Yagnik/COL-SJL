"""
Simple notification module for the MVP.

When an intrusion-like event is detected, this module prints a
human-readable alert to the console. In a real deployment you can
extend this to send SMS, email, or push notifications.
"""

from __future__ import annotations

from typing import Optional


class ConsoleNotifier:
    """
    Console-based notifier used for development and early testing.
    """

    def __init__(self) -> None:
        pass

    def notify_intrusion_probability(
        self,
        *,
        video_path: str,
        probability: float,
        threshold: float,
    ) -> None:
        """
        Model-driven intrusion notification.

        IMPORTANT: per requirements, intrusion decisions come ONLY from the ML model.
        """
        print(f"Video: {video_path} -> Intrusion probability: {probability:.2f}")
        if probability > threshold:
            print("=== SECURITY ALERT ===")
            print(f"INTRUSION DETECTED (p={probability:.2f}, threshold={threshold:.2f})")
            print("======================")

