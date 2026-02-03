import argparse
import json
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.inference.predictor import build_detector, resolve_device
from src.utils.config import DEFAULT_CONFIG_PATH, load_config


def _parse_args():
    parser = argparse.ArgumentParser(description="Run RAID benchmark evaluation")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file",
    )
    parser.add_argument("--model-path", default=None, help="Path to trained model")
    parser.add_argument("--split", default="train", choices=["train", "extra", "test"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows")
    parser.add_argument("--no-adversarial", action="store_true")
    parser.add_argument("--output", default=None, help="Write predictions JSON to path")
    parser.add_argument("--eval-output", default=None, help="Write evaluation JSON to path")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    inference_cfg = config.get("inference", {})

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.environ.setdefault("RAID_CACHE_DIR", os.path.join(project_root, ".raid_cache"))

    from raid import run_detection, run_evaluation
    from raid.utils import load_data

    device, device_name = resolve_device()
    print(f"Device: {device.type} ({device_name})")

    model_path = os.path.abspath(
        args.model_path or inference_cfg.get("model_path", "outputs/best_model")
    )
    batch_size = int(args.batch_size or inference_cfg.get("batch_size", 16))
    max_length = int(args.max_length or inference_cfg.get("max_length", 256))

    detector = build_detector(model_path, device, batch_size, max_length)

    df = load_data(split=args.split, include_adversarial=not args.no_adversarial)
    if args.limit:
        df = df.head(args.limit)

    predictions = run_detection(detector, df)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(predictions, f)

    if args.split in {"train", "extra"} and not args.skip_eval:
        results = run_evaluation(predictions, df)
        if args.eval_output:
            payload = {"results": results, "created_utc": datetime.utcnow().isoformat() + "Z"}
            with open(args.eval_output, "w", encoding="utf-8") as f:
                json.dump(payload, f)
        else:
            print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
