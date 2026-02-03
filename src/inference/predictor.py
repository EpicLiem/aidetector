from typing import Callable, Iterable, List, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_device() -> Tuple[torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return device, device_name


def load_model_and_tokenizer(
    model_path: str, device: torch.device
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()
    return model, tokenizer


def score_texts(
    texts: Iterable[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> List[float]:
    scores: List[float] = []
    batch = []
    for text in texts:
        batch.append(text if isinstance(text, str) else "")
        if len(batch) >= batch_size:
            scores.extend(_score_batch(batch, model, tokenizer, device, max_length))
            batch = []
    if batch:
        scores.extend(_score_batch(batch, model, tokenizer, device, max_length))
    return scores


def build_detector(
    model_path: str, device: torch.device, batch_size: int, max_length: int
) -> Callable[[List[str]], List[float]]:
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    def detector(texts: List[str]) -> List[float]:
        return score_texts(texts, model, tokenizer, device, batch_size, max_length)

    return detector


def _score_batch(
    batch: List[str],
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    max_length: int,
) -> List[float]:
    enc = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().tolist()
    return probs
