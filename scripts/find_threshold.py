import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.raid_bench import RaidBenchConfig, load_raid_bench
from src.utils.config import DEFAULT_CONFIG_PATH, load_config


def _parse_args():
    parser = argparse.ArgumentParser(description="Find a decision threshold on the val split")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file",
    )
    parser.add_argument("--output", default="threshold.json", help="Write threshold JSON to path")
    parser.add_argument("--metric", choices=["f1", "fpr"], default="f1")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--max-batches", type=int, default=200)
    return parser.parse_args()


def _score_batch(model, batch, device):
    inputs = {k: v.to(device) for k, v in batch.items() if k not in {"label", "idx"}}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
    return probs


def _collect_scores(model, dataloader, device, max_batches: int | None):
    scores = []
    labels = []
    for batch_idx, batch in enumerate(dataloader):
        labels.append(batch["label"].numpy())
        scores.append(_score_batch(model, batch, device))
        if max_batches is not None and (batch_idx + 1) >= max_batches:
            break
    return np.concatenate(scores), np.concatenate(labels)


def _best_f1_threshold(scores: np.ndarray, labels: np.ndarray):
    thresholds = np.linspace(0.0, 1.0, 101)
    best = {"threshold": 0.5, "f1": -1.0, "precision": 0.0, "recall": 0.0}
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best["f1"]:
            best = {"threshold": float(t), "f1": f1, "precision": precision, "recall": recall}
    return best


def _threshold_for_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float):
    neg_scores = scores[labels == 0]
    if len(neg_scores) == 0:
        return {"threshold": 1.0, "fpr": 0.0}
    threshold = float(np.quantile(neg_scores, 1.0 - target_fpr))
    preds = (scores >= threshold).astype(int)
    fp = int(((preds == 1) & (labels == 0)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"threshold": threshold, "fpr": fpr}


def main():
    args = _parse_args()
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_name"], use_fast=True)

    raid_cfg = RaidBenchConfig(
        name_or_path=dataset_cfg["name_or_path"],
        split_train=dataset_cfg["split_train"],
        split_val=dataset_cfg["split_val"],
        split_test=dataset_cfg["split_test"],
        text_column=dataset_cfg["text_column"],
        label_column=dataset_cfg["label_column"],
        label_from_column=dataset_cfg.get("label_from_column"),
        positive_values=dataset_cfg.get("positive_values"),
        negative_values=dataset_cfg.get("negative_values"),
        max_length=int(dataset_cfg.get("max_length", 256)),
        include_adversarial=bool(dataset_cfg.get("include_adversarial", True)),
        val_from_train=bool(dataset_cfg.get("val_from_train", True)),
        val_fraction=float(dataset_cfg.get("val_fraction", 0.1)),
        val_seed=int(dataset_cfg.get("val_seed", 42)),
        load_test=bool(dataset_cfg.get("load_test", False)),
        max_rows=dataset_cfg.get("max_rows"),
    )

    _, val_ds, _ = load_raid_bench(raid_cfg, tokenizer=tokenizer)
    if val_ds is None:
        raise ValueError("Validation split is required to find a threshold.")

    model_path = os.path.abspath(config["training"]["output_dir"] + "/best_model")
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    model.eval()

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
    scores, labels = _collect_scores(model, val_loader, device, args.max_batches)

    if args.metric == "f1":
        result = _best_f1_threshold(scores, labels)
    else:
        result = _threshold_for_fpr(scores, labels, args.target_fpr)

    payload = {"threshold": result["threshold"], "metric": args.metric, **result}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
