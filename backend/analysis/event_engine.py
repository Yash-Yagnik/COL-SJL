from typing import Any, Dict, Iterable, List
from time import gmtime, strftime


class EventAnalysisEngine:
    """
    Event analysis engine that consumes low-level tracking events and
    infers higher-level security events (possible intrusion, vehicle
    arrival / departure) suitable for report generation.
    """

    def _format_timestamp(self, seconds_from_start: float) -> str:
        """
        Convert a floating-point seconds value into a human-readable
        12‑hour clock string like ``"02:41 AM"``.
        """
        return strftime("%I:%M %p", gmtime(seconds_from_start))

    def infer_high_level_events(
        self, tracking_events: Iterable[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Inspect the stream of tracking events to infer higher-level
        events such as:

        - possible_intrusion
        - vehicle_arrival
        - vehicle_departure

        Each returned event has the structure:

        {
          "event_type": "possible_intrusion",
          "suspect_count": 3,
          "vehicles_detected": ["vehicle"],
          "timestamp": "02:41 AM"
        }
        """
        events_list: List[Dict[str, Any]] = list(tracking_events)

        high_level_events: List[Dict[str, Any]] = []

        seen_person_ids: set[int] = set()
        prev_vehicles_present = False
        intrusion_recorded = False

        for ev in events_list:
            timestamp_sec = float(ev.get("timestamp", 0.0))
            ts_str = self._format_timestamp(timestamp_sec)

            # Collect current frame people and vehicles.
            frame_person_ids: set[int] = set()
            frame_vehicles: List[str] = []

            for obj in ev.get("objects", []):
                if obj.get("type") == "person":
                    frame_person_ids.add(int(obj["id"]))
                elif obj.get("type") == "vehicle":
                    # For MVP we only know "vehicle" type,
                    # so we use a generic label.
                    frame_vehicles.append("vehicle")

            # Update global sets for suspects and current vehicle presence.
            seen_person_ids.update(frame_person_ids)
            vehicles_present = len(frame_vehicles) > 0

            # 1) Possible intrusion: first time we see any suspect.
            if not intrusion_recorded and len(seen_person_ids) > 0:
                intrusion_recorded = True
                high_level_events.append(
                    {
                        "event_type": "possible_intrusion",
                        "suspect_count": len(seen_person_ids),
                        "vehicles_detected": frame_vehicles,
                        "timestamp": ts_str,
                    }
                )

            # 2) Vehicle arrival: transition from no vehicles to some vehicles.
            if not prev_vehicles_present and vehicles_present:
                high_level_events.append(
                    {
                        "event_type": "vehicle_arrival",
                        "suspect_count": len(seen_person_ids),
                        "vehicles_detected": frame_vehicles,
                        "timestamp": ts_str,
                    }
                )

            # 3) Vehicle departure: transition from some vehicles to none.
            if prev_vehicles_present and not vehicles_present:
                high_level_events.append(
                    {
                        "event_type": "vehicle_departure",
                        "suspect_count": len(seen_person_ids),
                        "vehicles_detected": [],
                        "timestamp": ts_str,
                    }
                )

            prev_vehicles_present = vehicles_present

        return high_level_events

    def analyze_events(self, tracking_events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Consumes a sequence of tracking events and returns:

        - Overall intrusion flags (is_intrusion, suspect_count, vehicles_present)
        - The original per-frame tracking events
        - A list of inferred high-level events
        """
        all_events: List[Dict[str, Any]] = list(tracking_events)

        # Track unique person IDs and whether any vehicles were observed.
        person_ids: set[int] = set()
        any_vehicles = False

        for ev in all_events:
            for obj in ev.get("objects", []):
                if obj.get("type") == "person":
                    person_ids.add(int(obj["id"]))
                elif obj.get("type") == "vehicle":
                    any_vehicles = True

        suspect_count = len(person_ids)
        is_intrusion = suspect_count > 0

        high_level_events = self.infer_high_level_events(all_events)

        return {
            "is_intrusion": is_intrusion,
            "suspect_count": suspect_count,
            "vehicles_present": any_vehicles,
            "events": all_events,
            "high_level_events": high_level_events,
        }

