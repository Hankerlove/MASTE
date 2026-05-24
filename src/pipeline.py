"""
MASTE: Multi-Agent pipeline for Aspect Sentiment Triplet Extraction.

Pipeline stages:
  Stage 1 — Aspect Extraction Agent
  Stage 2 — Opinion Extraction Agent  (conditioned on aspects)
  Stage 3 — Sentiment Classification Agent  (conditioned on aspect-opinion pairs)
  Stage 4 — Consistency Check Agent  (halluci-detection + revision)

Each stage can be disabled independently for ablation studies.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from src.agents.aspect_agent import extract_aspects
from src.agents.opinion_agent import extract_opinions
from src.agents.sentiment_agent import classify_sentiments
from src.agents.consistency_agent import check_and_revise

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    model: str = "gpt-4o"
    temperature: float = 0.0
    # Ablation flags — set to False to disable a stage
    use_opinion_agent: bool = True
    use_sentiment_agent: bool = True
    use_consistency_agent: bool = True
    # If False, opinion_agent skips and uses empty string for opinion
    # (falls back to combined aspect-sentiment prompt in sentiment agent)


def run_pipeline(
    sentence: str,
    config: PipelineConfig,
) -> List[Tuple[str, str, str]]:
    """
    Run the full MASTE pipeline on a single sentence.

    Returns list of (aspect, opinion, sentiment) triplets.
    """
    # --- Stage 1: Aspect Extraction ---
    aspects = extract_aspects(
        sentence, model=config.model, temperature=config.temperature
    )
    logger.debug(f"Aspects: {aspects}")

    if not aspects:
        logger.warning("Stage1→∅ no aspects found | sent: %s", sentence[:80])
        return []

    # --- Stage 2: Opinion Extraction ---
    if config.use_opinion_agent:
        # Returns List[Tuple[str, str]]: one (aspect, opinion) pair per opinion span.
        # A single aspect may produce multiple pairs when it has multiple opinions.
        # Pairs with opinions not found verbatim in the sentence are already
        # filtered inside extract_opinions, so no further cleaning is needed here.
        pairs = extract_opinions(
            sentence, aspects, model=config.model, temperature=config.temperature
        )
        logger.debug(f"Opinion pairs: {pairs}")
    else:
        # Ablation: pass all aspects with empty opinion so Stage 3 / Stage 4
        # can still reason about sentiment from full sentence context.
        pairs = [(a, "") for a in aspects]

    if not pairs:
        logger.warning("Stage2→∅ no opinions found | aspects=%s | sent: %s", aspects, sentence[:80])
        return []

    # --- Stage 3: Sentiment Classification ---
    if config.use_sentiment_agent:
        triplets = classify_sentiments(
            sentence, pairs, model=config.model, temperature=config.temperature
        )
        logger.debug(f"Triplets after sentiment: {triplets}")
    else:
        # Ablation: default NEU
        triplets = [(a, o, "NEU") for a, o in pairs]

    # --- Stage 4: Consistency Check ---
    if config.use_consistency_agent and triplets:
        triplets = check_and_revise(
            sentence, triplets, model=config.model, temperature=config.temperature
        )
        logger.debug(f"Triplets after consistency check: {triplets}")

    return triplets


def run_pipeline_batch(
    sentences: List[str],
    config: PipelineConfig,
    verbose: bool = False,
) -> List[List[Tuple[str, str, str]]]:
    """
    Run pipeline on a batch of sentences.
    Returns list of triplet lists (one per sentence).
    """
    from tqdm import tqdm

    results = []
    for sent in tqdm(sentences, disable=not verbose, desc="Running MASTE"):
        try:
            triplets = run_pipeline(sent, config)
        except Exception as e:
            logger.error(f"Error on sentence '{sent[:60]}': {e}")
            triplets = []
        results.append(triplets)
    return results
