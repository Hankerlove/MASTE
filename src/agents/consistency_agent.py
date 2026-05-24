"""
Agent 4: Consistency Check Agent.

Reviews the full set of extracted triplets for internal consistency and
validates that aspect/opinion spans actually appear in the sentence.

Strictly zero-shot by default; the optional ``icl_messages`` argument
exists only for ablation studies.
"""

import json
import re
from typing import List, Optional, Tuple
from src.llm_client import chat_complete

SYSTEM_PROMPT = """You are a quality-control expert for aspect-based sentiment analysis.

Your task is to review a set of extracted (aspect, opinion, sentiment) triplets
and apply the following six checks in order:

1. HALLUCINATION — Remove any triplet whose aspect or opinion text does not appear
   verbatim (case-insensitive) anywhere in the sentence.
2. SPAN TRIM — If an opinion span contains unnecessary degree adverbs
   (very, extremely, really, quite, absolutely, definitely, a bit, too) that do NOT
   change the sentiment polarity, trim the opinion to its minimal core word.
   Examples:  "very clean"         → "clean"
              "definitely enjoyed" → "enjoyed"
              "absolutely gorgeous"→ "gorgeous"
              "a bit bland"        → "bland"
   Exception: Keep negation words intact — they DO change polarity.
   Examples:  "not bad"            → keep "not bad"
              "never disappoints"  → keep "never disappoints"
3. SENTIMENT ERROR — Fix the sentiment label if it does not correctly reflect
   the opinion word in the sentence context (consider negation, sarcasm, contrast).
4. MISSING — Add any aspect-opinion pair that is clearly and explicitly expressed
   in the sentence but absent from the proposed list.
5. DUPLICATION — Collapse exact-duplicate or semantically redundant triplets.

Rules:
- DEFAULT TO KEEPING triplets. Only remove for genuine hallucinations (check 1).
- When in doubt, KEEP rather than remove.
- Do NOT add speculative triplets not supported by the sentence text.
"""

# Optional handcrafted ICL demonstration. Not used in the canonical
# zero-shot MASTE pipeline; available for ablation only. This example
# would simultaneously demonstrate hallucination removal ("ambiance")
# and degree-adverb trimming ("very slow" -> "slow").
ICL_DEMO = [
    {
        "role": "user",
        "content": (
            'Sentence: "The food was delicious but the service was slow."\n'
            'Proposed triplets:\n'
            '1. aspect="food", opinion="delicious", sentiment="POS"\n'
            '2. aspect="service", opinion="very slow", sentiment="NEG"  '
            "(opinion span has unnecessary degree adverb)\n"
            '3. aspect="ambiance", opinion="great", sentiment="POS"  '
            "(\"ambiance\" does not appear in the sentence)\n\n"
            "Apply all checks and return only valid, corrected triplets as a JSON list: "
            '[{"aspect": ..., "opinion": ..., "sentiment": ...}]'
        ),
    },
    {
        "role": "assistant",
        "content": (
            '[{"aspect": "food", "opinion": "delicious", "sentiment": "POS"}, '
            '{"aspect": "service", "opinion": "slow", "sentiment": "NEG"}]'
        ),
    },
]


def check_and_revise(
    sentence: str,
    triplets: List[Tuple[str, str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    icl_messages: Optional[List[dict]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Review and revise the set of extracted triplets.

    Returns a cleaned list of (aspect, opinion, sentiment) triples.
    """
    if not triplets:
        return []

    triplets_str = "\n".join(
        f'{i+1}. aspect="{a}", opinion="{o}", sentiment="{s}"'
        for i, (a, o, s) in enumerate(triplets)
    )
    user_msg = (
        f'Sentence: "{sentence}"\n'
        f'Proposed triplets:\n{triplets_str}\n\n'
        "Review for hallucinations, sentiment errors, duplicates, and missing pairs. "
        "Return only valid, corrected triplets as a JSON list: "
        '[{"aspect": ..., "opinion": ..., "sentiment": ...}]'
    )
    icl = icl_messages or []
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(icl)
        + [{"role": "user", "content": user_msg}]
    )

    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=512)
    revised = _parse_triplets(raw)
    if not revised and triplets:
        return triplets
    return revised


def _parse_triplets(text: str) -> List[Tuple[str, str, str]]:
    """Parse JSON list of triplets from LLM output."""
    text = text.strip()
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                triplets = []
                for item in result:
                    if isinstance(item, dict):
                        a = str(item.get("aspect", "")).strip()
                        o = str(item.get("opinion", "")).strip()
                        s = str(item.get("sentiment", "NEU")).upper().strip()
                        if s not in ("POS", "NEG", "NEU"):
                            s = "NEU"
                        if a and o:
                            triplets.append((a, o, s))
                return triplets
        except json.JSONDecodeError:
            pass
    return []
