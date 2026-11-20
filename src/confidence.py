"""
Confidence estimation methods for generated answers.
"""

import numpy as np
import torch
from typing import List, Dict, Any


def max_token_probability(scores: List[torch.Tensor]) -> float:
    """Average of maximum token probabilities (simple confidence)."""
    if not scores:
        return 0.0
    probs = []
    for logit in scores:
        if logit is None:
            continue
        p = torch.softmax(logit, dim=-1)
        probs.append(p.max().item())
    return float(np.mean(probs)) if probs else 0.0


def sequence_logprob_confidence(logprobs: List[float]) -> float:
    """Convert average log-probability to a 0-1 confidence-like score."""
    if not logprobs:
        return 0.0
    avg_logprob = np.mean(logprobs)
    # Map typical logprob range to [0,1] (rough heuristic)
    conf = 1 / (1 + np.exp(-avg_logprob))  # sigmoid of average logprob
    return float(np.clip(conf, 0.0, 1.0))


def entropy_confidence(scores: List[torch.Tensor]) -> float:
    """Lower entropy → higher confidence."""
    if not scores:
        return 0.0
    ents = []
    for logit in scores:
        p = torch.softmax(logit, dim=-1)
        ent = -torch.sum(p * torch.log(p + 1e-12)).item()
        ents.append(ent)
    avg_ent = np.mean(ents)
    # Normalise roughly (vocab entropy can be large)
    conf = 1 / (1 + avg_ent / 5.0)
    return float(np.clip(conf, 0.0, 1.0))


def get_confidence(method: str, generation_outputs: Dict[str, Any]) -> float:
    """
    Unified interface.
    generation_outputs should contain whatever the generator provides
    (scores, logprobs, etc.)
    """
    if method == "max_prob":
        return max_token_probability(generation_outputs.get("scores", []))
    elif method == "entropy":
        return entropy_confidence(generation_outputs.get("scores", []))
    else:
        # Fallback
        return generation_outputs.get("confidence", 0.5)