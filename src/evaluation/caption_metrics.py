"""Dependency-free caption metrics for small human-reference evaluation sets."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens: Sequence[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1)))


def corpus_bleu(references: Sequence[Sequence[str]], hypotheses: Sequence[str], max_order: int = 4) -> float:
    """Smoothed corpus BLEU-4 on whitespace/punctuation-normalized tokens, in [0, 1]."""
    if len(references) != len(hypotheses) or not hypotheses:
        raise ValueError("BLEU requires the same non-zero number of reference sets and hypotheses")
    matches = [0] * max_order
    totals = [0] * max_order
    hyp_length = 0
    ref_length = 0
    for refs, hyp in zip(references, hypotheses):
        hyp_tokens = _tokens(hyp)
        ref_tokens = [_tokens(r) for r in refs]
        if not ref_tokens:
            raise ValueError("Each hypothesis must have at least one human reference")
        hyp_length += len(hyp_tokens)
        ref_length += min((len(ref) for ref in ref_tokens), key=lambda n: (abs(n - len(hyp_tokens)), n))
        for order in range(1, max_order + 1):
            hyp_counts = _ngrams(hyp_tokens, order)
            max_ref_counts: Counter[tuple[str, ...]] = Counter()
            for ref in ref_tokens:
                ref_counts = _ngrams(ref, order)
                for gram, count in ref_counts.items():
                    max_ref_counts[gram] = max(max_ref_counts[gram], count)
            matches[order - 1] += sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())
            totals[order - 1] += sum(hyp_counts.values())
    precisions = [(matches[i] + 1) / (totals[i] + 1) for i in range(max_order)]
    brevity_penalty = 1.0 if hyp_length > ref_length else math.exp(1 - ref_length / max(hyp_length, 1))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_order)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            current.append(previous[j - 1] + 1 if token_a == token_b else max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(reference: str, hypothesis: str) -> float:
    ref, hyp = _tokens(reference), _tokens(hypothesis)
    if not ref or not hyp:
        return 0.0
    common = _lcs_length(ref, hyp)
    precision, recall = common / len(hyp), common / len(ref)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def score_captions(references: Sequence[Sequence[str]], hypotheses: Sequence[str]) -> dict[str, float | int]:
    """Compute corpus BLEU and average best-reference ROUGE-L F1."""
    if len(references) != len(hypotheses) or not hypotheses:
        raise ValueError("Caption metrics require matched, non-empty references and hypotheses")
    best_rouge = [max(rouge_l_f1(ref, hyp) for ref in refs) for refs, hyp in zip(references, hypotheses)]
    return {
        "examples": len(hypotheses),
        "corpus_bleu_4": corpus_bleu(references, hypotheses),
        "mean_best_reference_rouge_l_f1": sum(best_rouge) / len(best_rouge),
    }
