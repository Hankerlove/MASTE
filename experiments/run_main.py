"""
Main experiment runner for MASTE.

Methods:
    zero_shot   - single-call ASTE prompt, no in-context examples (k=0).
    few_shot    - alias for direct with --icl_k (default 1).
    direct      - single-call direct prompt; --icl_k controls k (default 1).
    cot         - single-call chain-of-thought; --icl_k controls k (default 0).
    maste       - the proposed multi-agent pipeline. Always zero-shot:
                  every agent prompt contains zero in-context examples.

With --tag main_gpt35 and no --output_dir, results go under
results/main_gpt35/<condition>/ automatically.

Examples:
    # Few-shot baseline (1-shot direct prompting)
    python experiments/run_main.py --method direct --model openai/gpt-3.5-turbo

    # Pure zero-shot direct prompting
    python experiments/run_main.py --method zero_shot --model openai/gpt-3.5-turbo

    # Zero-shot CoT (default)
    python experiments/run_main.py --method cot --model openai/gpt-3.5-turbo

    # CoT + few-shot (k=1)
    python experiments/run_main.py --method cot --icl_k 1 --model openai/gpt-3.5-turbo

    # MASTE
    python experiments/run_main.py --method maste --model openai/gpt-3.5-turbo
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_aste
from src.evaluate import compute_f1, compute_per_domain
from src.pipeline import PipelineConfig, run_pipeline_batch
from src.baselines import (
    DIRECT_ICL_BY_K,
    COT_ICL_BY_K,
    chain_of_thought,
    direct,
    zero_shot,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = Path(__file__).parent.parent / "data" / "aste" / "SemEval-Triplet-data" / "ASTE-Data-V2-EMNLP2020"
RESULTS_DIR = Path(__file__).parent.parent / "results"
MAIN_GPT35_ROOT = RESULTS_DIR / "main_gpt35"


def resolve_method(method: str) -> str:
    """Normalize CLI method name to an internal baseline key."""
    if method == "few_shot":
        return "direct"
    return method


def default_output_dir(method: str, icl_k, tag: str) -> Optional[Path]:
    """Stable subdirs for the GPT-3.5 main experiment suite."""
    if tag != "main_gpt35":
        return None
    if method == "few_shot":
        k = 1 if icl_k is None else icl_k
        return MAIN_GPT35_ROOT / f"few_shot_k{k}"
    m = resolve_method(method)
    if m == "zero_shot":
        return MAIN_GPT35_ROOT / "zero_shot"
    if m == "maste":
        return MAIN_GPT35_ROOT / "maste"
    if m == "direct":
        k = 1 if icl_k is None else icl_k
        return MAIN_GPT35_ROOT / f"few_shot_k{k}"
    if m == "cot":
        k = 0 if icl_k is None else icl_k
        return MAIN_GPT35_ROOT / f"cot_k{k}"
    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["maste", "direct", "few_shot", "cot", "zero_shot"],
        default="maste",
        help="Method to run (few_shot is direct with k>=1 ICL)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="LLM identifier passed to the OpenAI-compatible API",
    )
    parser.add_argument(
        "--domains",
        nargs="+",
        default=["14res", "14lap", "15res", "16res"],
        help="Domains to evaluate",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["test", "dev"],
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Max examples per domain (for quick testing)",
    )
    parser.add_argument(
        "--icl_k",
        type=int,
        default=None,
        help=(
            "Number of in-context demonstrations for the single-call baselines. "
            "Defaults: direct=1, cot=0. Ignored by zero_shot and maste."
        ),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional tag (e.g. 'main_gpt35') used to organise output and "
        "to label rows in the API usage log.",
    )
    parser.add_argument(
        "--skip_completed_domains",
        action="store_true",
        help="Skip domains whose <domain>.json already exists in output_dir",
    )
    return parser.parse_args()


def build_baseline_runner(method: str, model: str, icl_k):
    """Return a function ``run(sentence) -> List[Tuple[str,str,str]]``."""
    if method == "zero_shot":
        if icl_k not in (None, 0):
            raise ValueError("zero_shot does not accept --icl_k > 0")
        return lambda s: zero_shot(s, model=model)

    if method == "direct":
        k = 1 if icl_k is None else icl_k
        if k not in DIRECT_ICL_BY_K:
            raise ValueError(f"Unsupported --icl_k={k} for direct; allowed: {list(DIRECT_ICL_BY_K)}")
        icl = DIRECT_ICL_BY_K[k]
        return lambda s: direct(s, model=model, icl_messages=icl)

    if method == "cot":
        k = 0 if icl_k is None else icl_k
        if k not in COT_ICL_BY_K:
            raise ValueError(f"Unsupported --icl_k={k} for cot; allowed: {list(COT_ICL_BY_K)}")
        icl = COT_ICL_BY_K[k]
        return lambda s: chain_of_thought(s, model=model, icl_messages=icl)

    raise ValueError(f"Unknown method: {method}")


def run_baseline_batch(sentences, runner, label: str):
    from tqdm import tqdm

    results = []
    for sent in tqdm(sentences, desc=f"Running {label}"):
        try:
            results.append(runner(sent))
        except Exception as e:  # noqa: BLE001 - we want any error logged but the run continues
            logger.error(f"Error: {e}")
            results.append([])
    return results


def main():
    args = parse_args()
    internal_method = resolve_method(args.method)

    if args.method == "few_shot" and args.icl_k is not None and args.icl_k < 1:
        raise ValueError("few_shot requires --icl_k >= 1")

    # Surface the run tag to the API usage log so cost analysis can group calls.
    if args.tag:
        os.environ["MASTE_USAGE_TAG"] = args.tag

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        stable = default_output_dir(args.method, args.icl_k, args.tag)
        if stable is not None:
            out_dir = stable
        else:
            suffix_parts = [internal_method, args.model.replace("/", "_")]
            if internal_method in ("direct", "cot") and args.icl_k is not None:
                suffix_parts.append(f"k{args.icl_k}")
            suffix_parts.append(timestamp)
            out_dir = RESULTS_DIR / "_".join(suffix_parts)
    out_dir.mkdir(parents=True, exist_ok=True)

    record_method = args.method if args.method == "few_shot" else internal_method
    effective_icl_k = args.icl_k
    if args.method == "few_shot" and effective_icl_k is None:
        effective_icl_k = 1

    logger.info(
        "Method=%s | Model=%s | Split=%s | icl_k=%s | tag=%s",
        record_method,
        args.model,
        args.split,
        effective_icl_k,
        args.tag,
    )
    logger.info("Domains: %s", args.domains)
    logger.info("Output:  %s", out_dir)
    logger.info("API base URL: %s", os.environ.get("OPENAI_BASE_URL", "(default)"))

    all_predictions = {}
    all_golds = {}

    if internal_method == "maste":
        config = PipelineConfig(model=args.model)
    else:
        runner = build_baseline_runner(internal_method, args.model, effective_icl_k)

    for domain in args.domains:
        logger.info("\n%s", "=" * 40)
        logger.info("Domain: %s", domain)

        domain_path = out_dir / f"{domain}.json"
        if args.skip_completed_domains and domain_path.exists():
            logger.info("  Skipping (already exists): %s", domain_path)
            with open(domain_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            all_predictions[domain] = [
                ex["pred"] for ex in cached.get("examples", [])
            ]
            all_golds[domain] = [
                ex["gold"] for ex in cached.get("examples", [])
            ]
            metrics = cached.get("metrics", {})
            logger.info(
                "  %s (cached): P=%s  R=%s  F1=%s",
                domain,
                metrics.get("precision"),
                metrics.get("recall"),
                metrics.get("f1"),
            )
            continue

        examples = load_aste(str(DATA_ROOT), domain, args.split)
        if args.max_examples:
            examples = examples[: args.max_examples]

        sentences = [ex.sentence for ex in examples]
        golds = [ex.get_triplet_tuples() for ex in examples]

        logger.info("  Loaded %d examples", len(examples))

        if internal_method == "maste":
            predictions = run_pipeline_batch(sentences, config, verbose=True)
        else:
            predictions = run_baseline_batch(sentences, runner, record_method)

        all_predictions[domain] = predictions
        all_golds[domain] = golds

        metrics = compute_f1(predictions, golds)
        logger.info(
            "  %s: P=%s  R=%s  F1=%s",
            domain,
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
        )

        domain_out = {
            "method": record_method,
            "model": args.model,
            "icl_k": effective_icl_k,
            "domain": domain,
            "metrics": metrics,
            "examples": [
                {"sentence": s, "gold": g, "pred": p}
                for s, g, p in zip(sentences, golds, predictions)
            ],
        }
        with open(out_dir / f"{domain}.json", "w") as f:
            json.dump(domain_out, f, indent=2, ensure_ascii=False)

    summary = compute_per_domain(all_predictions, all_golds)
    logger.info("\n%s", "=" * 40)
    logger.info("SUMMARY:")
    for domain, metrics in summary.items():
        if domain == "macro":
            logger.info("  MACRO AVG F1: %s", metrics["f1"])
        else:
            logger.info(
                "  %s: P=%s  R=%s  F1=%s",
                domain,
                metrics["precision"],
                metrics["recall"],
                metrics["f1"],
            )

    summary_out = {
        "method": record_method,
        "model": args.model,
        "icl_k": effective_icl_k,
        "split": args.split,
        "domains": args.domains,
        "tag": args.tag,
        "results": summary,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    logger.info("\nResults saved to: %s", out_dir)
    return summary


if __name__ == "__main__":
    main()
