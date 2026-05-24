"""
Evaluation metrics for ASTE.
Standard: exact-match triplet F1.

A predicted triplet is correct iff aspect span, opinion span,
AND sentiment ALL match exactly (case-insensitive, space-normalized).
"""

from typing import List, Tuple, Dict
from collections import defaultdict


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def compute_f1(
    predictions: List[List[Tuple[str, str, str]]],
    golds: List[List[Tuple[str, str, str]]],
) -> Dict[str, float]:
    """
    Compute micro-averaged precision, recall, F1 across all examples.

    Args:
        predictions: list of predicted triplet lists, one per example
        golds: list of gold triplet lists, one per example

    Returns:
        dict with keys: precision, recall, f1, n_pred, n_gold, n_correct
    """
    n_pred = n_gold = n_correct = 0
    for preds, golds_ex in zip(predictions, golds):
        pred_set = set(
            (normalize(a), normalize(o), s.upper())
            for a, o, s in preds
        )
        gold_set = set(
            (normalize(a), normalize(o), s.upper())
            for a, o, s in golds_ex
        )
        n_pred += len(pred_set)
        n_gold += len(gold_set)
        n_correct += len(pred_set & gold_set)

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_gold if n_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1": round(f1 * 100, 2),
        "n_pred": n_pred,
        "n_gold": n_gold,
        "n_correct": n_correct,
    }


def compute_per_domain(
    predictions: Dict[str, List[List[Tuple]]],
    golds: Dict[str, List[List[Tuple]]],
) -> Dict[str, Dict]:
    """Compute F1 per domain and macro-averaged."""
    results = {}
    for domain in predictions:
        results[domain] = compute_f1(predictions[domain], golds[domain])

    # Macro average F1 (average of per-domain F1s)
    macro_f1 = sum(r["f1"] for r in results.values()) / len(results)
    results["macro"] = {"f1": round(macro_f1, 2)}
    return results
