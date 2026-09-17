from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional

MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
DATE_PATTERNS = [
    re.compile(rf"\b(\d{{1,2}})\s+({'|'.join(MONTHS)})\s+(\d{{4}})\b", re.I),
    re.compile(rf"\b({'|'.join(MONTHS)})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.I),
    re.compile(r"\b(20\d{2})[-_/](\d{1,2})[-_/](\d{1,2})\b"),
    re.compile(r"\b(\d{1,2})[-_/](\d{1,2})[-_/](20\d{2})\b"),
]


def parse_date_from_text(text: str) -> Optional[date]:
    if not text:
        return None
    m = DATE_PATTERNS[0].search(text)
    if m:
        for fmt in ("%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", fmt).date()
            except ValueError:
                continue
    m = DATE_PATTERNS[1].search(text)
    if m:
        for fmt in ("%B %d %Y", "%b %d %Y"):
            try:
                return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", fmt).date()
            except ValueError:
                continue
    m = DATE_PATTERNS[2].search(text)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass
    m = DATE_PATTERNS[3].search(text)
    if m:
        try:
            return date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except ValueError:
            pass
    return None


def parse_date_from_filename(name: str) -> Optional[date]:
    stem = Path(name).stem
    return parse_date_from_text(stem) or parse_date_from_text(stem.replace("_", " "))
