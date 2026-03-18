"""
Canonical logp→prediction for rank labels.

Use deterministic tie-breaking: canonical label order, missing labels as -inf.
Matches the package rule so that student scripts and package helpers agree.
"""
from typing import Dict, Any, Optional

# Canonical order for tie-breaking (first in list wins when logp tie).
# Package uses ["exceptional", "strong", "fair", "limited"].
CANONICAL_RANK_LABELS = ("exceptional", "strong", "fair", "limited")

_NEG_INF = float("-inf")


def logp_argmax_canonical(logp: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Argmax over logp with canonical label-order tie-breaking.
    Missing labels are treated as -inf.
    Returns the winning label (lowercase) or None if logp is empty/invalid.
    """
    if not logp or not isinstance(logp, dict):
        return None
    best_label: Optional[str] = None
    best_score = _NEG_INF
    for label in CANONICAL_RANK_LABELS:
        raw = logp.get(label)
        try:
            score = float(raw) if raw is not None else _NEG_INF
        except (TypeError, ValueError):
            score = _NEG_INF
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def sorted_logp_pairs_canonical(logp: Optional[Dict[str, Any]]) -> list:
    """
    Return list of (label, score) sorted by score descending, with ties
    broken by canonical label order (first in CANONICAL_RANK_LABELS wins).
    Missing labels are treated as -inf and omitted from output.
    Used by validator to set top1_prediction, top2_prediction, etc.
    """
    if not logp or not isinstance(logp, dict):
        return []
    pairs: list = []
    for label in CANONICAL_RANK_LABELS:
        raw = logp.get(label)
        if raw is None:
            continue
        try:
            score = float(raw)
        except (TypeError, ValueError):
            continue
        pairs.append((label, score))
    # Sort by score desc, then by canonical index asc (first label wins tie)
    order_idx = {l: i for i, l in enumerate(CANONICAL_RANK_LABELS)}
    pairs.sort(key=lambda x: (x[1], -order_idx.get(x[0], 999)), reverse=True)
    # For reverse=True, (score, -idx) desc means higher score first; same score then lower -idx first (earlier label)
    # Actually: we want same score → first in canonical order. So key (score, -idx): higher score better; if same score, higher -idx better = smaller idx = earlier. So (score, -idx) reverse=True gives (high score, then high -idx = early label). Good.
    return pairs
