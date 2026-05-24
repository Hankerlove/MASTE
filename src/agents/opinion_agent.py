"""
Agent 2: Opinion Extraction Agent.

Extracts opinion expressions conditioned on identified aspect terms.
Supports one-to-many: a single aspect may map to multiple independent
opinion spans, each returned as a separate (aspect, opinion) pair.

Strictly zero-shot by default; the optional ``icl_messages`` argument
exists only for ablation studies.
"""

import json
import re
from typing import List, Optional, Tuple
from src.llm_client import chat_complete

SYSTEM_PROMPT = """You are an expert in aspect-based sentiment analysis.
Your task is to identify OPINION EXPRESSIONS for given aspect terms in a sentence.

An opinion expression is the SHORTEST word or phrase copied VERBATIM from the sentence
that captures the reviewer's evaluation of a specific aspect.
Copy the span exactly as it appears in the text, even if it contains misspellings or typos.

Rules:
- Extract the MINIMAL span. Do NOT include degree adverbs (very, extremely, really,
  quite, absolutely, definitely, a bit, too) unless they are part of an inseparable
  expression that changes meaning (e.g. keep "not bad", keep "not worth the price").
  Examples of minimal extraction:
    "very clean"           → extract "clean"
    "definitely enjoyed"   → extract "enjoyed"
    "absolutely gorgeous"  → extract "gorgeous"
    "not bad"              → keep "not bad"  (negation changes polarity)
- If an aspect has MULTIPLE independent opinion expressions in the sentence, output
  ONE entry per opinion — do NOT merge them into one string.
- Opinion expressions are not limited to adjectives: verbs expressing attitude or
  recommendation and nouns indicating preference are also valid opinion expressions when they
  are the clearest signal of the reviewer's stance toward the aspect.
- Omit aspects for which no opinion is found.
- Return a JSON list: [{"aspect": ..., "opinion": ...}]
"""

# Optional handcrafted ICL demonstration. Not used in the canonical
# zero-shot MASTE pipeline; available for ablation only.
ICL_DEMO = [
    {
        "role": "user",
        "content": (
            "Sentence: The food was delicious but the service was slow.\n"
            "Aspects: [\"food\", \"service\"]\n"
            "For each aspect, find its opinion expression(s). "
            "Return JSON list: [{\"aspect\": ..., \"opinion\": ...}]"
        ),
    },
    {
        "role": "assistant",
        "content": (
            '[{"aspect": "food", "opinion": "delicious"}, '
            '{"aspect": "service", "opinion": "slow"}]'
        ),
    },
]


def extract_opinions(
    sentence: str,
    aspects: List[str],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    icl_messages: Optional[List[dict]] = None,
) -> List[Tuple[str, str]]:
    """
    For each aspect, extract opinion expression(s) from the sentence.

    Returns a list of (aspect, opinion) pairs. One aspect may appear
    multiple times when it has more than one independent opinion span.
    Pairs whose opinion does not appear verbatim in the sentence are
    dropped.
    """
    if not aspects:
        return []

    aspects_str = json.dumps(aspects)
    user_msg = (
        f"Sentence: {sentence}\n"
        f"Aspects: {aspects_str}\n"
        "For each aspect find its opinion expression(s). "
        "Use the minimal span - no degree adverbs unless they change polarity. "
        "If one aspect has multiple opinions, output one entry per opinion. "
        "Omit aspects with no opinion. "
        "Return JSON list: [{\"aspect\": ..., \"opinion\": ...}]"
    )
    icl = icl_messages or []
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(icl)
        + [{"role": "user", "content": user_msg}]
    )

    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=512)
    return _parse_pairs(raw, aspects, sentence)


def _parse_pairs(
    text: str,
    aspects: List[str],
    sentence: str,
) -> List[Tuple[str, str]]:
    """
    Parse a JSON list of {aspect, opinion} objects from LLM output.

    Validation:
    - aspect must be (case-insensitively) one of the given aspects.
    - opinion must appear verbatim (case-insensitive) somewhere in the sentence.
    """
    text = text.strip()
    # Accept both list [...] and dict {...} wrappings for robustness
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                aspect_lower_map = {a.lower(): a for a in aspects}
                sentence_lower = sentence.lower()
                pairs: List[Tuple[str, str]] = []
                seen: set = set()
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    a_raw = str(item.get("aspect", "")).strip()
                    o_raw = str(item.get("opinion", "")).strip()
                    if not a_raw or not o_raw:
                        continue
                    # Resolve to original casing from the given aspects list
                    canonical_a = aspect_lower_map.get(a_raw.lower())
                    if canonical_a is None:
                        continue
                    # Opinion must actually appear in the sentence
                    if o_raw.lower() not in sentence_lower:
                        continue
                    key = (canonical_a.lower(), o_raw.lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    pairs.append((canonical_a, o_raw))
                return pairs
        except json.JSONDecodeError:
            pass
    return []
