import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training.trainer import train_entry
from src.utils.config import DEFAULT_CONFIG_PATH, load_config


def _parse_args():
    parser = argparse.ArgumentParser(description="Train RAID-bench BERT classifier")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to config YAML file",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    train_entry(config)


if __name__ == "__main__":
    main()
