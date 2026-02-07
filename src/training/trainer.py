import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.raid_bench import RaidBenchConfig, load_raid_bench
from src.models.bert_model import build_bert_classifier
from src.training.mining import (
    DistributedWeightedSampler,
    build_sample_weights,
    build_weighted_sampler,
    mine_hard_negatives,
    mine_hard_negatives_distributed,
)
from src.utils.metrics import compute_metrics


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(training_cfg: Dict) -> Tuple[torch.device, str, bool]:
    preferred = str(training_cfg.get("device", "auto")).lower()
    use_xla = False
    device = torch.device("cpu")
    device_name = "cpu"

    if preferred in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(0)
        else:
            print("Requested CUDA but it is not available. Falling back to CPU.")
        return device, device_name, use_xla

    if preferred == "xla":
        try:
            import torch_xla.core.xla_model as xm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Requested XLA device but torch_xla is not installed."
            ) from exc
        device = xm.xla_device()
        device_name = "xla"
        use_xla = True
        return device, device_name, use_xla

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        return device, device_name, use_xla

    try:
        import torch_xla.core.xla_model as xm
    except ModuleNotFoundError:
        return device, device_name, use_xla

    device = xm.xla_device()
    device_name = "xla"
    use_xla = True
    return device, device_name, use_xla


def _build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    sampler,
    training_cfg: Dict,
    device: torch.device,
) -> DataLoader:
    num_workers = int(training_cfg.get("dataloader_num_workers", 0))
    pin_memory = bool(training_cfg.get("pin_memory", device.type == "cuda"))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    prefetch_factor = training_cfg.get("prefetch_factor")
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    persistent_workers = training_cfg.get("persistent_workers")
    if num_workers > 0 and persistent_workers is not None:
        kwargs["persistent_workers"] = bool(persistent_workers)
    return DataLoader(dataset, **kwargs)


def _evaluate(
    model,
    dataloader,
    device,
    pos_label: int = 1,
    max_batches: int | None = None,
) -> Dict[str, float]:
    model.eval()
    preds = []
    labels = []
    pos_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            labels.append(batch["label"].numpy())
            inputs = {k: v.to(device) for k, v in batch.items() if k not in {"label", "idx"}}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            batch_preds = torch.argmax(probs, dim=-1).cpu().numpy()
            pos_probs.append(probs[:, pos_label].cpu().numpy())
            preds.append(batch_preds)
            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    pos_probs_arr = np.concatenate(pos_probs, axis=0)
    metrics = compute_metrics(preds, labels)
    metrics["label_pos_rate"] = float(labels.mean()) if len(labels) else 0.0
    metrics["pred_pos_rate"] = float(preds.mean()) if len(preds) else 0.0
    metrics["avg_pos_prob"] = float(pos_probs_arr.mean()) if len(pos_probs_arr) else 0.0
    metrics["pos_label"] = float(pos_label)
    return metrics


def _label_stats(dataset) -> tuple[float, Dict[int, int]] | None:
    if dataset is None:
        return None
    labels = np.array(dataset["label"])
    if len(labels) == 0:
        return 0.0, {}
    unique, counts = np.unique(labels, return_counts=True)
    counts_dict = {int(label): int(count) for label, count in zip(unique, counts)}
    return float(labels.mean()), counts_dict


def _is_master(use_xla: bool, xm) -> bool:
    if not use_xla or xm is None:
        return True
    return xm.is_master_ordinal()


def _train_main(
    config: Dict,
    rank: int | None = None,
    world_size: int | None = None,
    use_xla: bool = False,
    distributed: bool = False,
) -> Dict[str, float]:
    seed = int(config.get("seed", 42))
    _set_seed(seed)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    mining_cfg = config.get("mining", {})

    output_dir = training_cfg.get("output_dir", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    xm = None
    if use_xla:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        device_name = "xla"
    else:
        device, device_name, use_xla = _resolve_device(training_cfg)
    print(f"Device: {device.type} ({device_name})")
    if device.type == "cuda":
        print(f"CUDA available: {torch.cuda.is_available()} | GPUs: {torch.cuda.device_count()}")
    if use_xla:
        print("XLA available: True")

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
        map_num_proc=dataset_cfg.get("map_num_proc"),
        map_batch_size=dataset_cfg.get("map_batch_size"),
    )
    train_ds, val_ds, _ = load_raid_bench(raid_cfg, tokenizer=tokenizer)
    if _is_master(use_xla, xm):
        print(f"Train size: {len(train_ds)}")
        print(f"Val size: {len(val_ds) if val_ds is not None else 0}")
        print(f"Mining enabled: {mining_cfg.get('enabled', True)}")
    train_label_stats = _label_stats(train_ds)
    val_label_stats = _label_stats(val_ds)
    if _is_master(use_xla, xm):
        if train_label_stats is not None:
            train_pos_rate, train_counts = train_label_stats
            print(f"Train label pos rate: {train_pos_rate:.4f} | counts: {train_counts}")
        if val_label_stats is not None:
            val_pos_rate, val_counts = val_label_stats
            print(f"Val label pos rate: {val_pos_rate:.4f} | counts: {val_counts}")

    model = build_bert_classifier(
        pretrained_name=model_cfg["pretrained_name"],
        num_labels=int(model_cfg.get("num_labels", 2)),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )

    num_epochs = int(training_cfg.get("num_epochs", 3))
    batch_size = int(training_cfg.get("batch_size", 16))
    world_size_eff = int(world_size or 1)
    steps_per_epoch = max(1, len(train_ds) // (batch_size * world_size_eff))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * float(training_cfg.get("warmup_ratio", 0.1)))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    writer = None
    if _is_master(use_xla, xm):
        writer = SummaryWriter(log_dir=os.path.join(output_dir, "runs"))
        if train_label_stats is not None:
            writer.add_scalar("data/train_label_pos_rate", train_label_stats[0], 0)
        if val_label_stats is not None:
            writer.add_scalar("data/val_label_pos_rate", val_label_stats[0], 0)
    best_f1 = -1.0
    best_metrics = None

    hard_negative_ids = set()
    pos_label = int(mining_cfg.get("pos_label", 1))
    mining_enabled = bool(mining_cfg.get("enabled", True))

    if _is_master(use_xla, xm):
        print(f"Starting training: epochs={num_epochs}, batch_size={batch_size}")

    for epoch in range(1, num_epochs + 1):
        if mining_enabled and epoch >= int(mining_cfg.get("start_epoch", 1)):
            if (epoch - int(mining_cfg.get("start_epoch", 1))) % int(
                mining_cfg.get("mine_every_epochs", 1)
            ) == 0:
                mine_sampler = None
                if distributed:
                    mine_sampler = DistributedSampler(
                        train_ds,
                        num_replicas=world_size_eff,
                        rank=int(rank or 0),
                        shuffle=False,
                    )
                mine_loader = _build_dataloader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=mine_sampler is None,
                    sampler=mine_sampler,
                    training_cfg=training_cfg,
                    device=device,
                )
                if distributed and mine_sampler is not None and hasattr(mine_sampler, "set_epoch"):
                    mine_sampler.set_epoch(epoch)

                if use_xla:
                    from torch_xla.distributed.parallel_loader import ParallelLoader

                    mine_loader = ParallelLoader(mine_loader, [device]).per_device_loader(device)

                mine_iter = mine_loader
                if _is_master(use_xla, xm):
                    mine_iter = tqdm(
                        mine_loader,
                        desc=f"Mining epoch {epoch}",
                        total=len(mine_loader),
                        unit="batch",
                        file=sys.stdout,
                        dynamic_ncols=True,
                        mininterval=1.0,
                    )

                if distributed:
                    hard_negative_ids = mine_hard_negatives_distributed(
                        model,
                        mine_iter,
                        device=device,
                        xm=xm,
                        pos_label=pos_label,
                        top_fraction=float(mining_cfg.get("hard_negative_top_fraction", 0.1)),
                        tag=f"hard_negatives_epoch_{epoch}",
                    )
                else:
                    hard_negative_ids = mine_hard_negatives(
                        model,
                        mine_iter,
                        device=device,
                        pos_label=pos_label,
                        top_fraction=float(mining_cfg.get("hard_negative_top_fraction", 0.1)),
                    )
                if _is_master(use_xla, xm):
                    print(f"Hard negatives mined: {len(hard_negative_ids)}")
                    if writer is not None:
                        writer.add_scalar("mining/hard_negative_count", len(hard_negative_ids), epoch)

        sampler = None
        if mining_enabled and hard_negative_ids:
            if distributed:
                weights = build_sample_weights(
                    train_ds,
                    hard_negative_ids=hard_negative_ids,
                    hard_negative_weight=float(mining_cfg.get("hard_negative_weight", 3.0)),
                    pos_label=pos_label,
                )
                sampler = DistributedWeightedSampler(
                    weights=weights,
                    num_replicas=world_size_eff,
                    rank=int(rank or 0),
                    seed=seed,
                )
            else:
                sampler = build_weighted_sampler(
                    train_ds,
                    hard_negative_ids=hard_negative_ids,
                    hard_negative_weight=float(mining_cfg.get("hard_negative_weight", 3.0)),
                    pos_label=pos_label,
                )
        if distributed and sampler is None:
            sampler = DistributedSampler(
                train_ds,
                num_replicas=world_size_eff,
                rank=int(rank or 0),
                shuffle=True,
            )

        train_loader = _build_dataloader(
            train_ds,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            training_cfg=training_cfg,
            device=device,
        )
        if distributed and sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

        if use_xla:
            from torch_xla.distributed.parallel_loader import ParallelLoader

            train_loader = ParallelLoader(train_loader, [device]).per_device_loader(device)

        model.train()
        epoch_loss = 0.0
        batch_iter = train_loader
        if _is_master(use_xla, xm):
            batch_iter = tqdm(
                train_loader,
                desc=f"Epoch {epoch}",
                total=len(train_loader),
                unit="batch",
                file=sys.stdout,
                dynamic_ncols=True,
                mininterval=1.0,
            )

        for step, batch in enumerate(batch_iter):
            optimizer.zero_grad(set_to_none=True)
            inputs = {k: v.to(device) for k, v in batch.items() if k not in {"label", "idx"}}
            labels = batch["label"].to(device)

            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training_cfg.get("grad_clip_norm", 1.0))
            )
            if use_xla and xm is not None:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()
            scheduler.step()
            if use_xla and xm is not None:
                xm.mark_step()

            epoch_loss += loss.item()

            if (
                writer is not None
                and step % int(training_cfg.get("eval_every_steps", 200)) == 0
                and val_ds is not None
            ):
                val_loader = _build_dataloader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    sampler=None,
                    training_cfg=training_cfg,
                    device=device,
                )
                metrics = _evaluate(
                    model,
                    val_loader,
                    device=device,
                    pos_label=pos_label,
                    max_batches=training_cfg.get("val_max_batches"),
                )
                for k, v in metrics.items():
                    writer.add_scalar(f"val/{k}", v, epoch * steps_per_epoch + step)

        avg_loss = epoch_loss / max(1, len(train_loader))
        if writer is not None:
            writer.add_scalar("train/loss", avg_loss, epoch)

        if val_ds is not None and writer is not None:
            val_loader = _build_dataloader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                sampler=None,
                training_cfg=training_cfg,
                device=device,
            )
            metrics = _evaluate(
                model,
                val_loader,
                device=device,
                pos_label=pos_label,
                max_batches=training_cfg.get("val_max_batches"),
            )
            for k, v in metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_metrics = metrics
                model.save_pretrained(os.path.join(output_dir, "best_model"))
                tokenizer.save_pretrained(os.path.join(output_dir, "best_model"))

    if writer is not None:
        model.save_pretrained(os.path.join(output_dir, "final_model"))
        tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))
        writer.close()
    return {
        "f1": float(best_f1),
        "best_metrics": best_metrics or {},
        "best_model_path": os.path.join(output_dir, "best_model"),
    }


def _train_xla_worker(rank: int, config: Dict, world_size: int) -> None:
    _train_main(
        config,
        rank=rank,
        world_size=world_size,
        use_xla=True,
        distributed=True,
    )


def train_entry(config: Dict) -> Dict[str, float]:
    training_cfg = config.get("training", {})
    device = str(training_cfg.get("device", "auto")).lower()
    xla_distributed = bool(training_cfg.get("xla_distributed", False))

    if device == "xla" and xla_distributed:
        try:
            import torch_xla.distributed.xla_multiprocessing as xmp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torch_xla is required for XLA distributed.") from exc
        world_size = int(training_cfg.get("xla_cores", 8))
        xmp.spawn(
            _train_xla_worker,
            args=(config, world_size),
            nprocs=world_size,
            start_method="fork",
        )
        return {}

    use_xla = device == "xla"
    return _train_main(config, use_xla=use_xla, distributed=False)


def train(config: Dict) -> Dict[str, float]:
    training_cfg = config.get("training", {})
    use_xla = str(training_cfg.get("device", "auto")).lower() == "xla"
    return _train_main(config, use_xla=use_xla, distributed=False)
