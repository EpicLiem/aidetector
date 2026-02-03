import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.predictor import load_model_and_tokenizer, resolve_device, score_texts
from src.utils.config import DEFAULT_CONFIG_PATH, load_config


def _parse_args():
    parser = argparse.ArgumentParser(description="Run inference on sampled RAID train data")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file",
    )
    parser.add_argument("--model-path", default=None, help="Path to trained model")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--output", default="train_sample_predictions.json")
    parser.add_argument("--threshold", type=float, default=None, help="Score threshold for AI label")
    parser.add_argument("--threshold-path", default=None, help="Load threshold JSON from file")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    inference_cfg = config.get("inference", {})

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("RAID_CACHE_DIR", os.path.join(project_root, ".raid_cache"))

    from raid.utils import load_data

    df = load_data(split="train", include_adversarial=not args.no_adversarial)
    df = df.sample(n=args.sample_size, random_state=42)
    texts = df["generation"].tolist()

    model_path = os.path.abspath(
        args.model_path or inference_cfg.get("model_path", "outputs/best_model")
    )
    batch_size = int(args.batch_size or inference_cfg.get("batch_size", 16))
    max_length = int(args.max_length or inference_cfg.get("max_length", 256))
    threshold_path = args.threshold_path or inference_cfg.get("threshold_path")

    device, _ = resolve_device()
    model, tokenizer = load_model_and_tokenizer(model_path, device)
    scores = score_texts(texts, model, tokenizer, device, batch_size, max_length)
    threshold = args.threshold
    if threshold is None and threshold_path:
        with open(threshold_path, "r", encoding="utf-8") as f:
            threshold = json.load(f).get("threshold")

    payload = [
        {
            "id": row_id,
            "model": model,
            "domain": domain,
            "text": text,
            "score": score,
            "pred_label": "ai" if threshold is not None and score >= threshold else "human"
            if threshold is not None
            else None,
        }
        for row_id, model, domain, text, score in zip(
            df["id"].tolist(),
            df["model"].tolist(),
            df["domain"].tolist(),
            texts,
            scores,
        )
    ]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
