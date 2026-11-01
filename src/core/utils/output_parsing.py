"""
Utilities for parsing model outputs into reasoning + final answer.

We use this to support workflows where the model can output free-form reasoning
but the *final answer* must be exactly one label word.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple, Sequence, Dict, Any


DEFAULT_LABELS = ("exceptional", "strong", "fair", "limited")

# For deterministic tie-breaking: missing label in logp is treated as -inf.
NEG_INF = -float("inf")


def logp_to_top1_top2(
    logp: Dict[str, Any],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Tuple[Optional[str], Optional[float], Optional[str], Optional[float]]:
    """
    Argmax over logp using canonical label order. Ties are broken by first in
    canonical order. Missing labels are treated as -inf (excluded from top).

    Returns (top1_label, top1_logp, top2_label, top2_logp).
    """
    if not logp or not labels:
        return None, None, None, None
    # (label, logp_val) for each canonical label; missing -> -inf
    indexed = []
    for i, lab in enumerate(labels):
        v = logp.get(lab)
        val = float(v) if v is not None else NEG_INF
        indexed.append((lab, val, i))
    # Sort: higher logp first; tie-break by earlier in canonical order (smaller i)
    indexed.sort(key=lambda x: (x[1], -x[2]), reverse=True)
    if not indexed:
        return None, None, None, None
    t1_label, t1_val, _ = indexed[0]
    t1_logp = t1_val if t1_val != NEG_INF else None
    if len(indexed) < 2:
        return t1_label, t1_logp, None, None
    t2_label, t2_val, _ = indexed[1]
    t2_logp = t2_val if t2_val != NEG_INF else None
    return t1_label, t1_logp, t2_label, t2_logp


def normalize_label(
    text: Optional[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Optional[str]:
    """
    Normalize a text snippet to one of the allowed labels (lowercase).

    Strips whitespace, newlines, and markdown/punctuation symbols, then
    requires an exact match against the label set. No fuzzy or partial
    matching — ambiguous responses are rejected (return None).

    Methodology note:
        Responses were accepted only if they matched one of the valid tier
        notations after removing whitespace, punctuation, and markdown
        formatting symbols. All other responses were coded as incorrect.
    """
    if not text:
        return None
    t = str(text).strip()
    if not t:
        return None

    t = re.sub(r'[\*\_\#\`\.\,\!\?\n\r]', '', t).strip()
    if not t:
        return None

    label_set = {lab.lower(): lab.lower() for lab in labels}
    return label_set.get(t.lower())


def parse_reasoning_and_final(
    response_text: Optional[str],
    labels: Sequence[str] = DEFAULT_LABELS,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse model output into (reasoning_content, prediction_label).

    Supported formats:
    1) <reasoning>...</reasoning>\\nExceptional
    2) free-form reasoning ...\\nExceptional
    3) <final>Exceptional</final> (reasoning optional)

    Returns:
        reasoning_content: may be None if absent
        prediction_label: one of labels (lowercase) or None
    """
    if not response_text:
        return None, None

    text = str(response_text).strip()
    if not text:
        return None, None

    # Tag-based extraction
    # Support both <reasoning>...</reasoning> and <think>...</think> styles.
    m_reason = re.search(r"<reasoning>\s*([\s\S]*?)\s*</reasoning>", text, flags=re.IGNORECASE)
    if not m_reason:
        m_reason = re.search(r"<think>\s*([\s\S]*?)\s*</think>", text, flags=re.IGNORECASE)
    m_final_tag = re.search(r"<final>\s*([\s\S]*?)\s*</final>", text, flags=re.IGNORECASE)

    reasoning = m_reason.group(1).strip() if m_reason else None

    if m_final_tag:
        final_raw = m_final_tag.group(1).strip()
        pred = normalize_label(final_raw, labels=labels)
        return reasoning, pred

    # Otherwise: last non-empty line is the final answer word
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, None

    last = lines[-1]
    pred = normalize_label(last, labels=labels)
    if reasoning is None:
        reasoning = "\n".join(lines[:-1]).strip() or None

    return reasoning, pred


