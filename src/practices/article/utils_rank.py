"""
Rank alias normalization utilities.

Normalize various representations/synonyms to canonical labels:
  exceptional, strong, fair, limited
Returns lower-case canonical label or None if cannot map.
"""
from typing import Any, Optional


_CANONICAL = {"exceptional", "strong", "fair", "limited"}

_ALIASES = {
    "exceptional": {
        "exceptional", "excellent", "top", "tier1", "a", "a*", "best", "ex",
    },
    "strong": {
        "strong", "good", "tier2", "b", "above_average",
    },
    "fair": {
        "fair", "average", "tier3", "c", "moderate",
    },
    "limited": {
        "limited", "poor", "weak", "low", "tier4", "d",
    },
}

_NUMERIC_MAP = {
    4: "exceptional",
    3: "strong",
    2: "fair",
    1: "limited",
}


def normalize_rank(value: Any) -> Optional[str]:
    if value is None:
        return None
    # Enum value.name
    if hasattr(value, "name"):
        s = str(getattr(value, "name") or "").strip()
        if not s:
            return None
        s_low = s.lower()
        if s_low in _CANONICAL:
            return s_low
        # Enum names like "Exceptional" -> exceptional
        if s_low.capitalize() in {"Exceptional", "Strong", "Fair", "Limited"}:
            # already covered by s_low in _CANONICAL if lower-cased
            pass
    # Numeric mapping
    try:
        if isinstance(value, (int, float)) and int(value) in _NUMERIC_MAP:
            return _NUMERIC_MAP[int(value)]
    except Exception:
        pass
    # String mapping
    s = str(value or "").strip().lower()
    if not s:
        return None
    if s in _CANONICAL:
        return s
    for canon, aliases in _ALIASES.items():
        if s in aliases:
            return canon
    return None

