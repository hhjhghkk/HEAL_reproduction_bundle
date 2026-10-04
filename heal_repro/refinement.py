from __future__ import annotations

import re

from .data import parse_response


# Audited vocabulary equivalences observed between EAI object identifiers and
# natural task wording.  The mapping is intentionally small and explicit; it
# is not a general semantic-similarity model.
REFUSAL_REASON_EQUIVALENTS: dict[str, frozenset[str]] = {
    "light": frozenset({"lights"}),
    "keyboard": frozenset({"computer"}),
    "food_food": frozenset({"food", "groceries"}),
    "freezer": frozenset({"fridge", "refrigerator"}),
    "stereo": frozenset({"radio"}),
    "cup": frozenset({"glass", "water_glass"}),
    "novel": frozenset({"book", "textbook"}),
    "address_book": frozenset({"book"}),
    "oven": frozenset({"stove"}),
    "pot": frozenset({"pan", "frying_pan"}),
    "couch": frozenset({"sofa"}),
}


def _identifier_text(value: object) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return re.sub(r"_+", "_", text).strip("_")


def reason_match_kind(reason: str, missing_objects: list[str]) -> str:
    """Return ``exact``, ``equivalent`` or ``none`` for a refusal reason."""

    normalized_reason = f"_{_identifier_text(reason)}_"
    for missing in missing_objects:
        exact = _identifier_text(missing)
        if exact and f"_{exact}_" in normalized_reason:
            return "exact"
    for missing in missing_objects:
        for equivalent in REFUSAL_REASON_EQUIVALENTS.get(missing, frozenset()):
            normalized = _identifier_text(equivalent)
            if normalized and f"_{normalized}_" in normalized_reason:
                return "equivalent"
    return "none"


def validate_blocked_refinement(record: dict) -> dict:
    """Validate one model-generated refusal against its verifier evidence."""

    payload = parse_response(str(record.get("response", "")))
    parsed = isinstance(payload, dict)
    goals_empty = bool(
        parsed
        and payload.get("node goals") == []
        and payload.get("edge goals") == []
        and payload.get("action goals") == []
    )
    reason = str(payload.get("reason", "")).strip() if parsed else ""
    missing = [str(item) for item in record.get("missing_required_objects", [])]
    match_kind = reason_match_kind(reason, missing) if reason else "none"
    passed = parsed and goals_empty and bool(reason) and match_kind != "none"
    return {
        "passed": passed,
        "format_valid": parsed,
        "goals_empty": goals_empty,
        "reason_nonempty": bool(reason),
        "reason_match": match_kind,
        "reason": reason,
        "missing_required_objects": missing,
    }
