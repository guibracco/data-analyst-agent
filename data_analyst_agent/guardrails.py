"""Input guardrails for the data-analyst pipeline."""

import re

# Patterns that suggest a column may contain PII.
PII_PATTERNS = [
    r"ssn",
    r"social.?security",
    r"e.?mail",
    r"phone",
    r"credit.?card",
    r"password",
    r"passport",
    r"driver.?licen[sc]e",
    r"date.?of.?birth",
    r"dob",
    r"national.?id",
    r"tax.?id",
]

_PII_RE = re.compile("|".join(PII_PATTERNS), re.IGNORECASE)


def check_pii_columns(columns: list[str]) -> list[str]:
    """Return column names that match known PII patterns."""
    return [col for col in columns if _PII_RE.search(col)]
