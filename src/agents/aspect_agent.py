"""
Agent 1: Aspect Extraction Agent.

Identifies aspect terms (opinion targets) in a sentence.

This agent is strictly zero-shot by default: the prompt contains the
system instructions and the input sentence only. The optional
``icl_messages`` argument exists purely for ablation studies and is
documented as a deviation from the canonical MASTE setup.
"""

import json
import re
import logging
from typing import List, Optional
from src.llm_client import chat_complete

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert in aspect-based sentiment analysis.
Your task is to identify ASPECT TERMS in a sentence.

An aspect term is any word or phrase naming WHAT is being evaluated. It can be:
- a specific feature or attribute (e.g., "food", "battery life", "service", "price")
- the overall entity or experience (e.g., "place", "restaurant", "atmosphere", "experience")
- a specific dish, item or product (e.g., "sushi", "rolls", "Bison", "coffee")
- a person or role being evaluated (e.g., "staff", "waiter", "bartender", "manager")

Rules:
- Extract only aspects that are explicitly present in the sentence — do not infer or paraphrase.
- Copy the aspect span EXACTLY as it appears in the sentence, including any misspellings.
- Use the minimal span that identifies the evaluated target.
- An aspect can be a single word or a multi-word phrase.
- Return a JSON list of strings only. If no aspects are found, return [].
"""

# Optional handcrafted ICL demonstration (NOT drawn from any ASTE split).
# Not included in the canonical MASTE prompt; kept for ablation use only.
ICL_DEMO = [
    {
        "role": "user",
        "content": (
            "Sentence: The food was delicious but the service was slow.\n"
            "Identify all aspect terms. Return a JSON list of strings."
        ),
    },
    {
        "role": "assistant",
        "content": '["food", "service"]',
    },
]


def extract_aspects(
    sentence: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    icl_messages: Optional[List[dict]] = None,
) -> List[str]:
    """
    Extract aspect terms from a sentence.

    Args:
        sentence: input review sentence.
        model: LLM identifier passed to :func:`chat_complete`.
        temperature: sampling temperature (0.0 = deterministic).
        icl_messages: optional list of (user, assistant) demonstration
            messages. Default is no demonstrations (zero-shot).

    Returns:
        List of aspect term strings that appear verbatim in ``sentence``.
    """
    user_msg = (
        f"Sentence: {sentence}\n"
        "Identify all aspect terms. Return a JSON list of strings. "
        "If no aspects are found, return []."
    )
    icl = icl_messages or []
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + list(icl)
        + [{"role": "user", "content": user_msg}]
    )

    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=256)
    aspects = _parse_list(raw)

    # Span validation: only keep aspects that appear verbatim in the sentence.
    # This eliminates paraphrased or hallucinated spans that cause cascade errors
    # in downstream agents.
    sentence_lower = sentence.lower()
    validated = [a for a in aspects if a.lower() in sentence_lower]
    logger.debug(f"Aspect span validation: {aspects} → {validated}")
    return validated


def _parse_list(text: str) -> List[str]:
    """Parse JSON list from LLM output, with fallback."""
    text = text.strip()
    match = re.search(r'\[.*?\]', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(x).strip() for x in result if x]
        except json.JSONDecodeError:
            pass
    if "," in text:
        items = [x.strip().strip('"\'[]') for x in text.split(",")]
        return [x for x in items if x]
    if text and text != "[]":
        return [text.strip().strip('"\'[]')]
    return []
