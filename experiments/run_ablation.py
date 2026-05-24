"""
Ablation study for MASTE (GPT-3.5-turbo).

Tests contribution of each pipeline stage on 14res test (default), matching
the method section (Aspect → Opinion → Sentiment → Consistency).

Conditions (paper names in parentheses):
  full                    — S1+S2+S3+S4 (MASTE full)
  no_consistency          — w/o Consistency Agent (S4): no triplet-set verification
  no_opinion              — w/o Opinion Agent (S2): no span-aware / one-to-many extraction
  no_sentiment            — w/o Sentiment Agent (S3): all pairs labeled NEU
  no_opinion_no_consistency — w/o S2 and S4: aspect list → NEU-ish sentiment → no verifier

Usage:
    PYTHONPATH=. python experiments/run_ablation.py \\
        --model openai/gpt-3.5-turbo --domain 14res --split test \\
        --max_examples 0 --output_dir results/ablation/gpt35_14res
"""

from __future__ import annotations

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_aste
from src.evaluate import compute_f1
from src.pipeline import PipelineConfig, run_pipeline_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_ROOT = (
    Path(__file__).parent.parent
    / "data"
    / "aste"
    / "SemEval-Triplet-data"
    / "ASTE-Data-V2-EMNLP2020"
)
RESULTS_DIR = Path(__file__).parent.parent / "results" / "ablation"

# Paper Table order: full, w/o S4, w/o S2, w/o S3, w/o S2+S4
DEFAULT_CONDITION_ORDER = [
    "full",
    "no_consistency",
    "no_opinion",
    "no_sentiment",
    "no_opinion_no_consistency",
]

CONDITION_META: Dict[str, Dict[str, Any]] = {
    "full": {
        "paper_label": "MASTE (full)",
        "disabled_stages": [],
        "hypothesis": "Upper bound; all four agents active.",
    },
    "no_consistency": {
        "paper_label": "w/o Consistency (S4)",
        "disabled_stages": ["S4"],
        "hypothesis": "Precision drops (span bloat, hallucinations); recall may rise slightly.",
    },
    "no_opinion": {
        "paper_label": "w/o Opinion (S2)",
        "disabled_stages": ["S2"],
        "hypothesis": "Recall drops; loses minimal-span and one-to-many opinion pairing.",
    },
    "no_sentiment": {
        "paper_label": "w/o Sentiment (S3)",
        "disabled_stages": ["S3"],
        "hypothesis": "F1 collapses (all NEU labels).",
    },
    "no_opinion_no_consistency": {
        "paper_label": "w/o S2 + S4",
        "disabled_stages": ["S2", "S4"],
        "hypothesis": "Largest combined degradation.",
    },
}


def make_config(condition: str) -> PipelineConfig:
    if condition == "full":
        return PipelineConfig(
            use_opinion_agent=True,
            use_sentiment_agent=True,
            use_consistency_agent=True,
        )
    if condition == "no_opinion":
        return PipelineConfig(
            use_opinion_agent=False,
            use_sentiment_agent=True,
            use_consistency_agent=True,
        )
    if condition == "no_consistency":
        return PipelineConfig(
            use_opinion_agent=True,
            use_sentiment_agent=True,
            use_consistency_agent=False,
        )
    if condition == "no_sentiment":
        return PipelineConfig(
            use_opinion_agent=True,
            use_sentiment_agent=False,
            use_consistency_agent=True,
        )
    if condition == "no_opinion_no_consistency":
        return PipelineConfig(
            use_opinion_agent=False,
            use_sentiment_agent=True,
            use_consistency_agent=False,
        )
    raise ValueError(f"Unknown ablation condition: {condition}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-3.5-turbo")
    parser.add_argument(
        "--domain",
        default="14res",
        help="Domain to ablate on (or 'all')",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Cap examples per domain (omit or 0 for full split)",
    )
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tag", default="ablation_gpt35")
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITION_ORDER,
        help="Ablation conditions to run",
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip condition if <domain>_<condition>.json already exists",
    )
    return parser.parse_args()


def run_ablation_domain(domain: str, args, out_dir: Path) -> Dict[str, dict]:
    examples = load_aste(str(DATA_ROOT), domain, args.split)
    if args.max_examples is not None and args.max_examples > 0:
        examples = examples[: args.max_examples]

    sentences = [ex.sentence for ex in examples]
    golds = [ex.get_triplet_tuples() for ex in examples]

    logger.info("Domain %s: %d examples", domain, len(examples))

    domain_results: Dict[str, dict] = {}
    for cond_name in args.conditions:
        if cond_name not in CONDITION_META:
            logger.warning("Unknown condition: %s", cond_name)
            continue

        out_path = out_dir / f"{domain}_{cond_name}.json"
        if args.skip_completed and out_path.exists():
            logger.info("  Skipping %s (exists)", cond_name)
            with open(out_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            domain_results[cond_name] = cached["metrics"]
            continue

        config = make_config(cond_name)
        config.model = args.model

        logger.info("  Running: %s — %s", cond_name, CONDITION_META[cond_name]["paper_label"])
        predictions = run_pipeline_batch(sentences, config, verbose=True)
        metrics = compute_f1(predictions, golds)

        logger.info(
            "    P=%.2f  R=%.2f  F1=%.2f",
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )
        domain_results[cond_name] = metrics

        detail = {
            "condition": cond_name,
            "paper_label": CONDITION_META[cond_name]["paper_label"],
            "hypothesis": CONDITION_META[cond_name]["hypothesis"],
            "disabled_stages": CONDITION_META[cond_name]["disabled_stages"],
            "domain": domain,
            "model": args.model,
            "split": args.split,
            "metrics": metrics,
            "examples": [
                {"sentence": s, "gold": g, "pred": p}
                for s, g, p in zip(sentences, golds, predictions)
            ],
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(detail, fh, indent=2, ensure_ascii=False)

    return domain_results


def main():
    args = parse_args()
    if args.tag:
        os.environ["MASTE_USAGE_TAG"] = args.tag

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        model_slug = args.model.replace("/", "_")
        out_dir = RESULTS_DIR / f"ablation_{model_slug}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)

    domains = (
        ["14res", "14lap", "15res", "16res"] if args.domain == "all" else [args.domain]
    )

    all_results: Dict[str, Dict[str, dict]] = {}
    for domain in domains:
        all_results[domain] = run_ablation_domain(domain, args, out_dir)

    logger.info("\nAblation Summary (F1):")
    logger.info("%-32s %s", "Condition", "  ".join(f"{d:>8}" for d in domains))
    for cond in args.conditions:
        label = CONDITION_META.get(cond, {}).get("paper_label", cond)
        row = []
        for domain in domains:
            f1 = all_results.get(domain, {}).get(cond, {}).get("f1", "-")
            row.append(f"{f1:>8.2f}" if isinstance(f1, (int, float)) else f"{f1:>8}")
        logger.info("%-32s %s", label, "  ".join(row))

    summary = {
        "model": args.model,
        "split": args.split,
        "domains": domains,
        "condition_order": args.conditions,
        "condition_meta": CONDITION_META,
        "results": all_results,
    }
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("\nSaved to %s", out_dir)


if __name__ == "__main__":
    main()
