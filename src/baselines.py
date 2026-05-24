"""
Single-call LLM baselines for ASTE.

Prompts are deliberately minimal and structurally aligned with the
ASTE prompting protocol in MiniConGTS (Sun et al., 2024, Appendix A.6),
but rephrased so that any performance gap between MASTE and these
baselines is attributable to the prompting architecture, not wording.

Three single-call methods are provided, each available at k = 0, 1, 2
in-context demonstrations:

* ``zero_shot``      - direct prompt, k = 0 (no demonstrations).
* ``direct``         - direct prompt with k demonstrations (default k = 1
                       a.k.a. "few-shot" in the main table).
* ``chain_of_thought`` - reasoning prompt, default k = 0 (CoT zero-shot).
                       For "CoT + few-shot" call with ``icl_messages``
                       supplied, or use :func:`cot_few_shot`.

The handcrafted ICL examples below are NOT drawn from any ASTE training
or test split. They are written by the authors specifically to
illustrate format and to expose typical phenomena (multiple opinions per
aspect, degree-adverb minimisation):

* ICL #1 (restaurant): ``The food was delicious but the service was slow.``
  --> (food, delicious, POS), (service, slow, NEG).
* ICL #2 (laptop, one-to-many + degree-adverb minimisation):
  ``The keyboard is comfortable and very responsive.``
  --> (keyboard, comfortable, POS), (keyboard, responsive, POS).

MASTE itself is strictly zero-shot (no ICL inside any agent prompt) and
is implemented in :mod:`src.pipeline`. This module only powers the
single-call comparison baselines.
"""

import json
import re
from typing import List, Optional, Tuple

from src.llm_client import chat_complete

# ---------------------------------------------------------------------------
# Direct (no reasoning) prompts.
# Wording follows MiniConGTS Appendix A.6 Table 10 ("zero-shot" entry),
# adapted to require a JSON list output so that the parser is shared with
# the chain-of-thought variant.
# ---------------------------------------------------------------------------

DIRECT_SYSTEM = """You are a specialist in aspect-based sentiment analysis.
From the input sentence, extract every (aspect, opinion, sentiment) triplet.

Rules:
- Sentiment must be POS, NEG, or NEU.
- Aspect and opinion strings must be copied verbatim from the sentence (no paraphrase).

Respond with a JSON list only—no other text:
[{"aspect": ..., "opinion": ..., "sentiment": ...}]
"""

DIRECT_ICL_1 = [
    {
        "role": "user",
        "content": "Sentence: The food was delicious but the service was slow.",
    },
    {
        "role": "assistant",
        "content": (
            '[{"aspect": "food", "opinion": "delicious", "sentiment": "POS"}, '
            '{"aspect": "service", "opinion": "slow", "sentiment": "NEG"}]'
        ),
    },
]

DIRECT_ICL_2 = [
    {
        "role": "user",
        "content": "Sentence: The keyboard is comfortable and very responsive.",
    },
    {
        "role": "assistant",
        "content": (
            '[{"aspect": "keyboard", "opinion": "comfortable", "sentiment": "POS"}, '
            '{"aspect": "keyboard", "opinion": "responsive", "sentiment": "POS"}]'
        ),
    },
]

# k -> ordered list of demonstration messages.
DIRECT_ICL_BY_K = {
    0: [],
    1: DIRECT_ICL_1,
    2: DIRECT_ICL_1 + DIRECT_ICL_2,
}


def zero_shot(
    sentence: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> List[Tuple[str, str, str]]:
    """Single-call ASTE prompt with NO in-context examples (k = 0)."""
    return _direct_call(sentence, model=model, temperature=temperature, icl_messages=[])


def direct(
    sentence: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    icl_messages: Optional[List[dict]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Single-call direct prompting.

    Defaults to k = 1 in-context example (the canonical "few-shot" baseline
    in the main results table). Pass ``icl_messages=[]`` to force k = 0
    (equivalent to :func:`zero_shot`).
    """
    icl = DIRECT_ICL_1 if icl_messages is None else icl_messages
    return _direct_call(sentence, model=model, temperature=temperature, icl_messages=icl)


def few_shot(
    sentence: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    k: int = 1,
) -> List[Tuple[str, str, str]]:
    """Convenience wrapper: few-shot with k in {0, 1, 2}."""
    if k not in DIRECT_ICL_BY_K:
        raise ValueError(f"Unsupported few-shot k={k}; expected one of {list(DIRECT_ICL_BY_K)}")
    return _direct_call(
        sentence,
        model=model,
        temperature=temperature,
        icl_messages=DIRECT_ICL_BY_K[k],
    )


def _direct_call(
    sentence: str,
    model: str,
    temperature: float,
    icl_messages: List[dict],
) -> List[Tuple[str, str, str]]:
    messages = (
        [{"role": "system", "content": DIRECT_SYSTEM}]
        + list(icl_messages)
        + [{"role": "user", "content": f"Sentence: {sentence}"}]
    )
    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=512)
    return _parse_triplets(raw)


# ---------------------------------------------------------------------------
# Chain-of-Thought prompts.
# Wording follows MiniConGTS Appendix A.6 Table 10 ("CoT" entry), but the
# canonical setup here is purely zero-shot: the prompt contains the
# step-by-step instructions and definitions, with NO worked example.
# A worked example is added only for the "CoT + few-shot" condition.
# ---------------------------------------------------------------------------

COT_SYSTEM = """You are a specialist in aspect-based sentiment analysis.
Analyze the input sentence step by step, then emit the final triplets.

Terminology:
- Aspect: the evaluated target (typically a noun or noun phrase).
- Opinion: the wording that conveys attitude toward that aspect (often an adjective).
- Sentiment: POS, NEG, or NEU toward the aspect.

Procedure:
1. List every aspect explicitly mentioned in the sentence.
2. For each aspect, identify the linked opinion span and assign POS, NEG, or NEU.
3. When one aspect pairs with multiple opinions, output one triplet per pair.
4. Keep aspect and opinion spans verbatim from the sentence (no paraphrase).
5. After brief reasoning, print the final answer as a JSON list with keys
   "aspect", "opinion", and "sentiment".

Final output format (JSON list only at the end):
[{"aspect": ..., "opinion": ..., "sentiment": ...}]
"""

COT_ICL_1 = [
    {
        "role": "user",
        "content": "Sentence: The food was delicious but the service was slow.",
    },
    {
        "role": "assistant",
        "content": (
            "1) Aspects found: 'food', 'service'.\n"
            "2) Opinion links: food → 'delicious'; service → 'slow'.\n"
            "3) Labels: delicious → POS; slow → NEG.\n\n"
            '[{"aspect": "food", "opinion": "delicious", "sentiment": "POS"}, '
            '{"aspect": "service", "opinion": "slow", "sentiment": "NEG"}]'
        ),
    },
]

COT_ICL_2 = [
    {
        "role": "user",
        "content": "Sentence: The keyboard is comfortable and very responsive.",
    },
    {
        "role": "assistant",
        "content": (
            "1) Aspects found: 'keyboard'.\n"
            "2) Opinion links: keyboard → 'comfortable' and 'responsive' "
            "(omit degree adverb 'very' from the span).\n"
            "3) Labels: comfortable → POS; responsive → POS.\n\n"
            '[{"aspect": "keyboard", "opinion": "comfortable", "sentiment": "POS"}, '
            '{"aspect": "keyboard", "opinion": "responsive", "sentiment": "POS"}]'
        ),
    },
]

COT_ICL_BY_K = {
    0: [],
    1: COT_ICL_1,
    2: COT_ICL_1 + COT_ICL_2,
}


def chain_of_thought(
    sentence: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    icl_messages: Optional[List[dict]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Single-call chain-of-thought prompting.

    Default is k = 0 (zero-shot CoT). Pass ``icl_messages`` to add
    demonstrations, e.g. for the "CoT + few-shot" condition.
    """
    icl = [] if icl_messages is None else icl_messages
    messages = (
        [{"role": "system", "content": COT_SYSTEM}]
        + list(icl)
        + [{"role": "user", "content": f"Sentence: {sentence}"}]
    )
    raw = chat_complete(messages, model=model, temperature=temperature, max_tokens=1024)
    return _parse_triplets(raw)


def cot_few_shot(
    sentence: str,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    k: int = 1,
) -> List[Tuple[str, str, str]]:
    """Convenience wrapper: chain-of-thought with k in {0, 1, 2}."""
    if k not in COT_ICL_BY_K:
        raise ValueError(f"Unsupported CoT k={k}; expected one of {list(COT_ICL_BY_K)}")
    return chain_of_thought(
        sentence,
        model=model,
        temperature=temperature,
        icl_messages=COT_ICL_BY_K[k],
    )


# ---------------------------------------------------------------------------
# Shared output parser
# ---------------------------------------------------------------------------

def _parse_triplets(text: str) -> List[Tuple[str, str, str]]:
    """Parse a JSON list of triplets from LLM output, with light fallback."""
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        return []
    if not isinstance(result, list):
        return []
    triplets: List[Tuple[str, str, str]] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        a = str(item.get("aspect", "")).strip()
        o = str(item.get("opinion", "")).strip()
        s = str(item.get("sentiment", "NEU")).upper().strip()
        if s not in ("POS", "NEG", "NEU"):
            s = "NEU"
        if a and o:
            triplets.append((a, o, s))
    return triplets
