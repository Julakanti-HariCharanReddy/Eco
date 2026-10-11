"""
Guarded LLM Explainer (Project Eco).

The model only rewords audited engine output into plain English. It never computes.
Every number in its reply is checked against the numbers the deterministic
engines produced (100% numeric grounding). Any reply that fails the check, times
out, is refused or errors falls back to the deterministic template, so the
endpoint always returns a valid, grounded explanation.

Opt-in: the LLM path runs only when ANTHROPIC_API_KEY is set (or ECO_LLM_ENABLED=1
with another Anthropic credential source). Otherwise the template is used.

Environment:
    ECO_LLM_MODEL      default "claude-opus-5-5"
    ECO_LLM_TIMEOUT_S  default 8 seconds (no retries; on timeout the template is used)
"""

import json
import logging
import os
import re
from typing import Any, Iterable, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You explain home-retrofit results to UK homeowners and mortgage advisors.

You receive a JSON object of results already computed by audited calculation engines.
Rules:
- Do not calculate, estimate, round differently, convert units, or add up anything.
- Only quote numbers exactly as they appear in the JSON (thousands separators are fine).
- Never introduce a number that is not in the JSON, including percentages, years or prices.
- Explain what the user asked about first, then the most useful supporting facts.
- If fabric measures were added as prerequisites, say why in one sentence (fabric-first).
- These are indicative estimates, not a valuation or regulated mortgage advice.
- Use short Markdown bullet points with **bold** key figures. At most 170 words."""

_NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?")


def llm_enabled() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY")) or os.getenv("ECO_LLM_ENABLED") == "1"


def _flatten_numbers(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield float(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from _flatten_numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from _flatten_numbers(v)
    elif isinstance(value, str):
        for m in _NUMBER_RE.findall(value):
            parsed = _parse_number(m)
            if parsed is not None:
                yield parsed


def _parse_number(token: str):
    try:
        return float(token.replace(",", "").lstrip("+"))
    except ValueError:
        return None


def allowed_numbers(*sources: Any) -> Set[float]:
    """Every number the reply may quote, in the display forms people use for it."""
    allowed: Set[float] = set()
    for n in (n for s in sources for n in _flatten_numbers(s)):
        for v in (n, abs(n)):
            allowed.update({v, round(v), round(v, 1), round(v, 2)})
            # Ratios stored as fractions are often shown as percentages (0.25 -> 25%).
            if 0 < v <= 1:
                allowed.add(round(v * 100, 2))
    return allowed


def ungrounded_numbers(text: str, allowed: Set[float]) -> list:
    """Numbers in `text` that do not match any audited value."""
    bad = []
    for token in _NUMBER_RE.findall(text):
        n = _parse_number(token)
        if n is None:
            continue
        candidates = {n, abs(n), round(abs(n), 1), round(abs(n), 2)}
        if not candidates & allowed:
            bad.append(token)
    return bad


def explain(
    metrics: Any,
    user_query: str,
    template_text: str,
) -> Tuple[str, str, bool]:
    """
    Returns (explanation, source, numeric_grounding_passed).
    source is "llm" or "template".
    """
    if not llm_enabled():
        return template_text, "template", True

    payload = metrics.model_dump() if hasattr(metrics, "model_dump") else metrics

    try:
        import anthropic

        client = anthropic.Anthropic().with_options(
            timeout=float(os.getenv("ECO_LLM_TIMEOUT_S", "8")),
            max_retries=0,
        )
        response = client.beta.messages.create(
            model=os.getenv("ECO_LLM_MODEL", DEFAULT_MODEL),
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"<question>{user_query}</question>\n"
                        f"<audited_results>{json.dumps(payload, default=str)}</audited_results>"
                    ),
                }
            ],
        )
    except Exception as exc:  # network, auth, timeout: the template is always safe
        logger.warning("LLM explainer unavailable, using template: %s", exc)
        return template_text, "template", True

    if response.stop_reason == "refusal":
        return template_text, "template", True

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    if not text:
        return template_text, "template", True

    # Small integers are allowed for list ordering ("Phase 1") and "100%" assurance wording.
    allowed = allowed_numbers(payload, user_query, list(range(0, 11)), 100)
    bad = ungrounded_numbers(text, allowed)
    if bad:
        logger.warning("LLM reply failed numeric grounding %s; using template", bad)
        return template_text, "template", False

    return text, "llm", True
