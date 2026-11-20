import json
import os
from typing import List, Dict, Any
import numpy as np


def load_qa_data(path: str) -> List[Dict[str, Any]]:
    """Load evaluation QA pairs."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_documents(doc_dir: str) -> List[Dict[str, str]]:
    """Load all .txt and .md files from a directory."""
    docs = []
    if not os.path.exists(doc_dir):
        return docs
    for fname in os.listdir(doc_dir):
        if fname.endswith((".txt", ".md")):
            path = os.path.join(doc_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                docs.append({"id": fname, "text": text})
    return docs


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Simple overlapping character chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def expected_calibration_error(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if np.any(mask):
            bin_acc = correct[mask].mean()
            bin_conf = confidences[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.mean()
    return float(ece)


def accuracy_from_answers(predictions: List[str], ground_truths: List[str]) -> float:
    """Very simple exact / substring match accuracy (for demo purposes)."""
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        pred_clean = pred.lower().strip()
        gt_clean = gt.lower().strip()
        if gt_clean in pred_clean or pred_clean in gt_clean:
            correct += 1
    return correct / len(predictions) if predictions else 0.0