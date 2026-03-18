"""
Incident report generator.

Takes high-level event data from the analysis engine and converts it
into a concise, human-readable emergency report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class IncidentEvent:
    """
    Simple container for a high-level event, aligned with the requested
    JSON structure.
    """

    event_type: str
    suspect_count: int
    vehicles_detected: List[str]
    timestamp: str  # already formatted, e.g. "02:41 AM"

    @classmethod
    def from_dict(cls, data: Dict) -> "IncidentEvent":
        return cls(
            event_type=data.get("event_type", ""),
            suspect_count=int(data.get("suspect_count", 0)),
            vehicles_detected=list(data.get("vehicles_detected", [])),
            timestamp=str(data.get("timestamp", "")),
        )

    def to_dict(self) -> Dict:
        return {
            "event_type": self.event_type,
            "suspect_count": self.suspect_count,
            "vehicles_detected": self.vehicles_detected,
            "timestamp": self.timestamp,
        }


class IncidentReportGenerator:
    """
    Converts one or more high-level events into a concise incident
    report, both as structured data and as a formatted text block.
    """

    def build_event_object(self, event: Dict) -> Dict:
        """
        Normalise an event dictionary into the requested shape:

        {
          "event_type": "possible_intrusion",
          "suspect_count": 3,
          "vehicles_detected": ["white sedan"],
          "timestamp": "02:41 AM"
        }
        """
        incident_event = IncidentEvent.from_dict(event)
        return incident_event.to_dict()

    def select_primary_event(self, events: List[Dict]) -> Optional[Dict]:
        """
        Pick the most important event for the report. For an MVP we
        prioritise:

        1. high_risk_intrusion (new behavioral engine)
        2. attempted_entry
        3. approaching_entry
        4. possible_intrusion (legacy)
        5. vehicle_arrival
        3. any other event
        """
        if not events:
            return None

        # New engine emits an intrusion summary event with "event" field.
        for ev in events:
            if ev.get("event") == "high_risk_intrusion":
                # Normalize to an IncidentEvent-like dict.
                return {
                    "event_type": "high_risk_intrusion",
                    "suspect_count": len(ev.get("actors", [])) or int(ev.get("suspect_count", 0) or 0),
                    "vehicles_detected": ev.get("vehicles_detected", []),
                    "timestamp": ev.get("timestamp", ""),
                    "raw": ev,
                }

        for ev in events:
            if ev.get("event_type") == "attempted_entry":
                return ev

        for ev in events:
            if ev.get("event_type") == "approaching_entry":
                return ev

        # Look for a "possible_intrusion" event first.
        for ev in events:
            if ev.get("event_type") == "possible_intrusion":
                return ev

        # Fall back to a vehicle_arrival event.
        for ev in events:
            if ev.get("event_type") == "vehicle_arrival":
                return ev

        # Otherwise just use the first event.
        return events[0]

    def generate_structured_report(self, events: List[Dict]) -> Optional[Dict]:
        """
        Generate a structured incident report dictionary from a list
        of high-level events.
        """
        primary = self.select_primary_event(events)
        if not primary:
            return None

        incident_event = IncidentEvent.from_dict(primary)

        # For vehicles we just join the detected list for now.
        vehicle_desc = (
            ", ".join(incident_event.vehicles_detected)
            if incident_event.vehicles_detected
            else "None / Unknown"
        )

        # Headline mapping for common behavioral events.
        if incident_event.event_type in ("high_risk_intrusion", "possible_intrusion"):
            incident_type = "Possible Intrusion Detected"
        elif incident_event.event_type == "attempted_entry":
            incident_type = "Entry Attempt Detected"
        elif incident_event.event_type == "approaching_entry":
            incident_type = "Approaching Entry Detected"
        else:
            incident_type = incident_event.event_type.replace("_", " ").title()

        return {
            "incident_type": incident_type,
            "suspects_detected": incident_event.suspect_count,
            "vehicles_detected": vehicle_desc,
            "timestamp": incident_event.timestamp,
            "raw_event": incident_event.to_dict(),
        }

    def format_text_report(self, report: Dict) -> str:
        """
        Turn a structured report dictionary into a concise,
        human-readable text block.
        """
        return (
            f"{report['incident_type']}\n\n"
            f"Suspects Detected: {report['suspects_detected']} individuals\n"
            f"Vehicle Detected: {report['vehicles_detected']}\n"
            f"Time: {report['timestamp']}"
        )

