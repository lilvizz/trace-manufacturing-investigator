from dataclasses import dataclass, field
from datetime import date, datetime
import re

from app.investigation.models import EvidenceItem


@dataclass
class TimelineEvent:
    date: date
    description: str
    evidence_chunk_ids: list[str] = field(default_factory=list)
    # Provenance of the date itself. All optional with defaults, so older
    # positional/keyword construction keeps working.
    #   date_text  : the exact text the date was parsed from ("27-MAR-2014")
    #   precision  : "day", or "month" when the source gives only a month
    #                and year ("Sept 2001"); the stored day is then 1 and
    #                must never be displayed as a real day
    #   kind       : "record_issued" when the date is a record's own
    #                issue/header date ("Date: 27-MAR-2014"), "mentioned"
    #                when the text refers to something dated
    #   ambiguous  : True for numeric dates such as 03/04/2014 that could
    #                be read day-first or month-first (read day-first, as
    #                the corpus convention, and flagged)
    date_text: str = ""
    precision: str = "day"
    kind: str = "mentioned"
    ambiguous: bool = False


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


_MONTH_NAMES = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|"
    r"aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)

# Text immediately before a date that marks it as the record's own
# issue/header date rather than a reference to an event.
# The label must start a header field (line start, a wide column gap or a
# "|" separator), so "Install Date:" or "Due date" stay event dates.
_ISSUE_LABEL = re.compile(
    r"(?:^|\s{2,}|\|\s*)(?:effective\s+)?(?:date|dated|issued|issue\s+date|"
    r"revision\s+date|rev\.?\s+date|date\s+of\s+issue|date\s+submitted)\s*[:.]?\s*$",
    re.IGNORECASE,
)


class TimelineBuilder:
    """
    Builds a timeline only from dates explicitly present in evidence.

    Dates are never inferred from filenames, titles, surrounding events or
    a model. Every event keeps the exact date text it was parsed from, its
    precision (day, or month when only "Sept 2001" is written), whether it
    is a record's own issue date, and whether a numeric date is ambiguous.
    """

    # Day-precision patterns, in priority order.
    _DATE_PATTERNS = (
        re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),
        re.compile(
            r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{4})\b"
        ),
        re.compile(
            r"\b(\d{1,2})[- ]"
            r"(" + _MONTH_NAMES + r")\.?"
            r"[- ,]+(\d{4})\b",
            re.IGNORECASE,
        ),
        # "June 5, 2001"
        re.compile(
            r"\b(" + _MONTH_NAMES + r")\.? (\d{1,2}),? (\d{4})\b",
            re.IGNORECASE,
        ),
    )

    # Month precision: "Sept 2001", "June 2001". Only used for text not
    # already covered by a day-precision match.
    _MONTH_PATTERN = re.compile(
        r"\b(" + _MONTH_NAMES + r")\.?,? (\d{4})\b",
        re.IGNORECASE,
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
            for found in self._extract_dated_mentions(item.content):
                timeline.add_event(
                    TimelineEvent(
                        date=found["date"],
                        description=found["description"],
                        evidence_chunk_ids=[item.chunk_id],
                        date_text=found["date_text"],
                        precision=found["precision"],
                        kind=found["kind"],
                        ambiguous=found["ambiguous"],
                    )
                )

        timeline.sort()
        return timeline

    def _extract_events(
        self,
        content: str,
    ) -> list[tuple[date, str]]:
        """Backward-compatible view: (date, line) pairs."""
        return [
            (found["date"], found["description"])
            for found in self._extract_dated_mentions(content)
        ]

    def _extract_dated_mentions(self, content: str) -> list[dict]:
        events = []

        for line in content.splitlines():
            line = line.strip()

            if not line:
                continue

            taken: list[tuple[int, int]] = []
            found_in_line: list[dict] = []

            def overlaps(span):
                return any(span[0] < end and start < span[1] for start, end in taken)

            # Every day-precision date in the line (not just the first).
            for pattern in self._DATE_PATTERNS:
                for match in pattern.finditer(line):
                    if overlaps(match.span()):
                        continue
                    parsed = self._parse_match(match)
                    # Invalid calendar dates are ignored, but still mark the
                    # span so a month pattern cannot re-read part of it.
                    taken.append(match.span())
                    if parsed is None:
                        continue
                    found_in_line.append(
                        self._mention(line, match, parsed, "day", self._is_ambiguous(match))
                    )

            for match in self._MONTH_PATTERN.finditer(line):
                if overlaps(match.span()):
                    continue
                month_token = match.group(1)
                # Lowercase "may 2014" is far more likely the verb.
                if not month_token[:1].isupper():
                    continue
                month = self._MONTHS.get(month_token[:3].lower())
                try:
                    parsed = date(int(match.group(2)), month, 1) if month else None
                except ValueError:
                    parsed = None
                taken.append(match.span())
                if parsed is not None:
                    found_in_line.append(self._mention(line, match, parsed, "month", False))

            found_in_line.sort(key=lambda f: f["start"])
            events.extend(found_in_line)

        return events

    @staticmethod
    def _mention(line: str, match: re.Match, parsed: date, precision: str, ambiguous: bool) -> dict:
        preceding = line[: match.start()]
        kind = "record_issued" if _ISSUE_LABEL.search(preceding) else "mentioned"
        return {
            "date": parsed,
            "description": line,
            "date_text": match.group(0),
            "precision": precision,
            "kind": kind,
            "ambiguous": ambiguous,
            "start": match.start(),
        }

    @staticmethod
    def _is_ambiguous(match: re.Match) -> bool:
        groups = match.groups()
        if len(groups) != 3:
            return False
        first, second, third = groups
        if not first.isdigit() or len(first) == 4 or not second.isdigit():
            return False
        a, b = int(first), int(second)
        return a <= 12 and b <= 12 and a != b

    def _parse_match(self, match: re.Match) -> date | None:
        groups = match.groups()

        try:
            if len(groups) != 3:
                return None

            first, second, third = groups

            # YYYY-MM-DD ("June" is also four characters, so check digits)
            if len(first) == 4 and first.isdigit():
                return date(
                    int(first),
                    int(second),
                    int(third),
                )

            # Month D, YYYY
            if not first.isdigit():
                month = self._MONTHS.get(first[:3].lower())
                if month is None:
                    return None
                return date(int(third), month, int(second))

            # DD/MM/YYYY or DD-MM-YYYY
            if second.isdigit():
                return date(
                    int(third),
                    int(second),
                    int(first),
                )

            # DD-MMM-YYYY
            month = self._MONTHS.get(second[:3].lower())

            if month is None:
                return None

            return date(
                int(third),
                month,
                int(first),
            )

        except ValueError:
            return None
