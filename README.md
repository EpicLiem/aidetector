# AI Text Classifier (RAID-bench + BERT)

This project trains a BERT-based classifier on the RAID-bench dataset for AI-generated text detection. It uses self-hard-negative mining to reduce false positives by focusing on the hardest negative examples each epoch.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --config config/config.yaml
```

## RAID Benchmark

RAID is a large benchmark for AI-generated text detection. See the dataset and docs:
- https://raid-bench.xyz
- https://huggingface.co/datasets/liamdugan/raid

Run a quick benchmark on RAID train/extra:

```bash
python scripts/raid_benchmark.py --model-path outputs/best_model --split train --eval-output eval.json
```

## RAID Submission

Generate predictions for the RAID test split and a starter metadata file:

```bash
python scripts/raid_submit.py --model-path outputs/best_model --predictions predictions.json --metadata metadata.json
```

Edit `metadata.json` to match RAID's submission template before uploading.

## Notes

- The dataset configuration supports both Hugging Face dataset names and local paths.
- Self-hard-negative mining re-weights the most confusing negative samples to reduce false positives.
