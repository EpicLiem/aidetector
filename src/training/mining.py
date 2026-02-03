from typing import Iterable, Set

import torch
from torch.utils.data import WeightedRandomSampler


def mine_hard_negatives(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    pos_label: int = 1,
    top_fraction: float = 0.1,
) -> Set[int]:
    model.eval()
    scores = []
    total_negatives = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch["label"].to(device)
            idxs = batch["idx"].cpu().tolist()
            inputs = {k: v.to(device) for k, v in batch.items() if k not in {"label", "idx"}}

            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pos_probs = probs[:, pos_label].cpu().tolist()

            for prob, label, idx in zip(pos_probs, labels.cpu().tolist(), idxs):
                if label != pos_label:
                    scores.append((prob, idx))
                    total_negatives += 1

    if total_negatives == 0:
        return set()

    scores.sort(key=lambda x: x[0], reverse=True)
    top_k = max(1, int(total_negatives * top_fraction))
    return {idx for _, idx in scores[:top_k]}


def build_weighted_sampler(
    dataset,
    hard_negative_ids: Set[int],
    hard_negative_weight: float,
    pos_label: int = 1,
) -> WeightedRandomSampler:
    idxs = dataset["idx"]
    labels = dataset["label"]
    weights = []

    for idx, label in zip(idxs, labels):
        if label != pos_label and idx in hard_negative_ids:
            weights.append(hard_negative_weight)
        else:
            weights.append(1.0)

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
