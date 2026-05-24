"""
Agent 3: Sentiment Reasoning Agent.

For each (aspect, opinion) pair, reasons about the sentiment polarity.

Strictly zero-shot by default; the optional ``icl_messages`` argument
exists only for ablation studies.
"""

import json
import re
from typing import List, Optional, Tuple
from src.llm_client import chat_complete

SYSTEM_PROMPT = """You are an expert sentiment analyst.
Your task is to determine the SENTIMENT POLARITY for aspect-opinion pairs.

Sentiment labels:
- POS: Positive sentiment (good, excellent, great, loved, etc.)
- NEG: Negative sentiment (bad, terrible, awful, hated, slow, etc.)
- NEU: Neutral sentiment (okay, average, normal, standard, etc.)

Rules:
- Consider the context of the full sentence, not just the opinion word in isolation.
- Be careful with negation (e.g., "not bad" → POS, "not great" → NEG).
- Consider intensifiers and modifiers.
"""

# Optional handcrafted ICL demonstration. Not used in the canonical
# zero-shot MASTE pipeline; available for ablation only.
ICL_DEMO = [
    {
        "role": "user",
        "content": (
            'Sentence: "The food was delicious but the service was slow."\n'
            'Aspect-Opinion pairs:\n'
            '- aspect: "food", opinion: "delicious"\n'
            '- aspect: "service", opinion: "slow"\n'
            "Classify each as POS, NEG, or NEU. "
            'Return JSON list: [{"aspect": ..., "opinion": ..., "sentiment": ...}]'
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


def classify_sentiments(
    sentence: str,
    aspect_opinion_pairs: List[Tuple[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.0,
    icl_messages: Optional[List[dict]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Classify sentiment for each (aspect, opinion) pair.

    Returns list of (aspect, opinion, sentiment) triples.
    """
    if not aspect_opinion_pairs:
        return []

    pairs_str = "\n".join(
        f'- aspect: "{a}", opinion: "{o}"'
        for a, o in aspect_opinion_pairs
    )
    user_msg = (
        f'Sentence: "{sentence}"\n'
        f'Aspect-Opinion pairs:\n{pairs_str}\n'
        "Classify each as POS, NEG, or NEU. "
        'Return JSON list: [{"aspect": ..., "opinion": ..., "sentiment": ...}]'
    )
    icl = icl_messages or []
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(icl)
        + [{"role": "user", "content": user_msg}]
    )

    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=512)
    return _parse_triplets(raw, aspect_opinion_pairs)


def _parse_triplets(
    text: str,
    fallback_pairs: List[Tuple[str, str]],
) -> List[Tuple[str, str, str]]:
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
                        a = item.get("aspect", "")
                        o = item.get("opinion", "")
                        s = item.get("sentiment", "NEU").upper()
                        if s not in ("POS", "NEG", "NEU"):
                            s = "NEU"
                        triplets.append((str(a), str(o), s))
                return triplets
        except json.JSONDecodeError:
            pass
    return [(a, o, "NEU") for a, o in fallback_pairs]
