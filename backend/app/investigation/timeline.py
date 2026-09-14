from dataclasses import dataclass, field
from datetime import date, datetime
import re

from app.investigation.models import EvidenceItem


@dataclass
class TimelineEvent:
    date: date
    description: str
    evidence_chunk_ids: list[str] = field(default_factory=list)


@dataclass
class InvestigationTimeline:
    events: list[TimelineEvent] = field(default_factory=list)

    def add_event(self, event: TimelineEvent) -> None:
        existing = next(
            (
                item
                for item in self.events
                if item.date == event.date
                and item.description == event.description
            ),
            None,
        )

        if existing is None:
            self.events.append(event)
            return

        for chunk_id in event.evidence_chunk_ids:
            if chunk_id not in existing.evidence_chunk_ids:
                existing.evidence_chunk_ids.append(chunk_id)

    def sort(self) -> None:
        self.events.sort(key=lambda event: event.date)


class TimelineBuilder:
    """
    Builds a timeline only from dates explicitly present in evidence.

    This first implementation intentionally does not infer dates or
    generate event descriptions with an LLM.
    """

    _DATE_PATTERNS = (
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
        re.compile(
            r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"
        ),
        re.compile(
            r"\b(\d{1,2})[- ]"
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            r"[- ](\d{4})\b",
            re.IGNORECASE,
        ),
    )

    _MONTHS = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    def build(self, evidence: list[EvidenceItem]) -> InvestigationTimeline:
        timeline = InvestigationTimeline()

        for item in evidence:
            for event_date, description in self._extract_events(item.content):
                timeline.add_event(
                    TimelineEvent(
                        date=event_date,
                        description=description,
                        evidence_chunk_ids=[item.chunk_id],
                    )
                )

        timeline.sort()
        return timeline

    def _extract_events(
        self,
        content: str,
    ) -> list[tuple[date, str]]:
        events = []

        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            for pattern in self._DATE_PATTERNS:
                match = pattern.search(line)

                if not match:
                    continue

                parsed_date = self._parse_match(match)

                if parsed_date is not None:
                    events.append((parsed_date, line))
                    break

        return events

    def _parse_match(self, match: re.Match) -> date | None:
        groups = match.groups()

        try:
            if len(groups) != 3:
                return None

            first, second, third = groups

            # YYYY-MM-DD
            if len(first) == 4:
                return date(
                    int(first),
                    int(second),
                    int(third),
                )

            # DD/MM/YYYY or DD-MM-YYYY
            if second.isdigit():
                return date(
                    int(third),
                    int(second),
                    int(first),
                )

            # DD-MMM-YYYY
            month = self._MONTHS.get(second.lower())

            if month is None:
                return None

            return date(
                int(third),
                month,
                int(first),
            )

        except ValueError:
            return None