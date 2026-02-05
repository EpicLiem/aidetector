import math
from typing import Iterable, Iterator, Set

import torch
from torch.utils.data import Sampler
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


def mine_hard_negatives_distributed(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    xm,
    pos_label: int = 1,
    top_fraction: float = 0.1,
    tag: str = "hard_negatives",
) -> Set[int]:
    if xm is None:
        raise ValueError("xm is required for distributed mining.")

    model.eval()
    local_scores = []

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
                    local_scores.append((prob, idx))

    all_scores = xm.mesh_reduce(tag, local_scores, lambda x, y: x + y)
    hard_negative_ids: Set[int] = set()
    if xm.is_master_ordinal():
        if not all_scores:
            hard_negative_ids = set()
        else:
            all_scores.sort(key=lambda x: x[0], reverse=True)
            top_k = max(1, int(len(all_scores) * top_fraction))
            hard_negative_ids = {idx for _, idx in all_scores[:top_k]}

    hard_negative_ids = xm.broadcast_master(hard_negative_ids)
    return hard_negative_ids


def build_sample_weights(
    dataset,
    hard_negative_ids: Set[int],
    hard_negative_weight: float,
    pos_label: int = 1,
) -> torch.Tensor:
    idxs = dataset["idx"]
    labels = dataset["label"]
    weights = []

    for idx, label in zip(idxs, labels):
        if label != pos_label and idx in hard_negative_ids:
            weights.append(hard_negative_weight)
        else:
            weights.append(1.0)

    return torch.tensor(weights, dtype=torch.double)


def build_weighted_sampler(
    dataset,
    hard_negative_ids: Set[int],
    hard_negative_weight: float,
    pos_label: int = 1,
) -> WeightedRandomSampler:
    weights_tensor = build_sample_weights(
        dataset,
        hard_negative_ids=hard_negative_ids,
        hard_negative_weight=hard_negative_weight,
        pos_label=pos_label,
    )
    return WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights: torch.Tensor,
        num_replicas: int,
        rank: int,
        seed: int = 0,
        replacement: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.weights = weights
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.replacement = replacement
        self.drop_last = drop_last
        self.epoch = 0

        if drop_last:
            self.num_samples = len(self.weights) // self.num_replicas
        else:
            self.num_samples = int(math.ceil(len(self.weights) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)

        if self.replacement:
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=True, generator=generator
            ).tolist()
        else:
            indices = torch.multinomial(
                self.weights, self.total_size, replacement=False, generator=generator
            ).tolist()

        indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
