"""
ASTE-Data-V2 loader.

Format per line:
  sentence####[(aspect_indices, opinion_indices, 'POS'|'NEG'|'NEU'), ...]

Indices are 0-based word-level positions.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional


@dataclass
class Triplet:
    aspect_indices: List[int]
    opinion_indices: List[int]
    sentiment: str  # 'POS', 'NEG', 'NEU'

    def aspect_span(self, tokens: List[str]) -> str:
        return " ".join(tokens[i] for i in self.aspect_indices)

    def opinion_span(self, tokens: List[str]) -> str:
        return " ".join(tokens[i] for i in self.opinion_indices)


@dataclass
class ASTEExample:
    sentence: str
    tokens: List[str]
    triplets: List[Triplet]
    domain: str = ""
    split: str = ""

    def get_triplet_tuples(self) -> List[Tuple[str, str, str]]:
        """Return list of (aspect_text, opinion_text, sentiment) tuples."""
        return [
            (t.aspect_span(self.tokens), t.opinion_span(self.tokens), t.sentiment)
            for t in self.triplets
        ]


def parse_triplet_line(line: str) -> Optional[Tuple[str, List[Triplet]]]:
    """Parse a single line of ASTE data."""
    line = line.strip()
    if not line:
        return None
    if "####" not in line:
        return None

    parts = line.split("####")
    sentence = parts[0].strip()
    tokens = sentence.split()

    raw_triplets = ast.literal_eval(parts[1].strip())
    triplets = []
    for aspect_ids, opinion_ids, sentiment in raw_triplets:
        if isinstance(aspect_ids, int):
            aspect_ids = [aspect_ids]
        if isinstance(opinion_ids, int):
            opinion_ids = [opinion_ids]
        triplets.append(Triplet(
            aspect_indices=sorted(aspect_ids),
            opinion_indices=sorted(opinion_ids),
            sentiment=sentiment,
        ))
    return sentence, triplets


def load_aste(data_dir: str, domain: str, split: str) -> List[ASTEExample]:
    """Load ASTE examples from file."""
    path = Path(data_dir) / domain / f"{split}_triplets.txt"
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            result = parse_triplet_line(line)
            if result is None:
                continue
            sentence, triplets = result
            tokens = sentence.split()
            examples.append(ASTEExample(
                sentence=sentence,
                tokens=tokens,
                triplets=triplets,
                domain=domain,
                split=split,
            ))
    return examples


def load_all_domains(data_dir: str, split: str = "test") -> dict:
    """Load all four ASTE domains for a given split."""
    domains = ["14res", "14lap", "15res", "16res"]
    return {d: load_aste(data_dir, d, split) for d in domains}
