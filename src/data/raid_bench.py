from dataclasses import dataclass
import importlib
from typing import Any, Optional, Tuple

from transformers import AutoTokenizer


@dataclass
class RaidBenchConfig:
    name_or_path: str
    split_train: str
    split_val: str
    split_test: str
    text_column: str
    label_column: str
    label_from_column: Optional[str] = None
    positive_values: Optional[list] = None
    negative_values: Optional[list] = None
    max_length: int = 256
    include_adversarial: bool = True
    val_from_train: bool = True
    val_fraction: float = 0.1
    val_seed: int = 42
    load_test: bool = False
    max_rows: Optional[int] = None
    map_num_proc: Optional[int] = None
    map_batch_size: Optional[int] = None


def _get_split(dataset_dict: Any, name: str) -> Optional[Any]:
    if name in dataset_dict:
        return dataset_dict[name]
    return None


def _map_kwargs(
    map_num_proc: Optional[int],
    map_batch_size: Optional[int],
) -> dict:
    kwargs = {}
    if map_num_proc is not None:
        kwargs["num_proc"] = int(map_num_proc)
    if map_batch_size is not None:
        kwargs["batch_size"] = int(map_batch_size)
    return kwargs


def _ensure_label_column(
    dataset: Any,
    label_column: str,
    require_label: bool = True,
    label_from_column: Optional[str] = None,
    positive_values: Optional[list] = None,
    negative_values: Optional[list] = None,
    map_num_proc: Optional[int] = None,
    map_batch_size: Optional[int] = None,
) -> Any:
    if label_from_column and label_from_column in dataset.column_names:
        pos_set = set(positive_values or [])
        neg_set = set(negative_values or [])

        def map_label(batch):
            labels = []
            for value in batch[label_from_column]:
                if value in pos_set:
                    labels.append(1)
                elif value in neg_set:
                    labels.append(0)
                elif not require_label:
                    labels.append(-1)
                else:
                    raise ValueError(
                        f"Unmapped label value '{value}' in column '{label_from_column}'."
                    )
            return {"label": labels}

        dataset = dataset.map(
            map_label,
            batched=True,
            **_map_kwargs(map_num_proc, map_batch_size),
        )
        if dataset.features["label"].dtype == "string":
            dataset = dataset.class_encode_column("label")
        return dataset

    if label_column not in dataset.column_names:
        if label_from_column and label_from_column in dataset.column_names:
            pos_set = set(positive_values or [])
            neg_set = set(negative_values or [])

            def map_label(batch):
                labels = []
                for value in batch[label_from_column]:
                    if value in pos_set:
                        labels.append(1)
                    elif value in neg_set:
                        labels.append(0)
                    elif not require_label:
                        labels.append(-1)
                    else:
                        raise ValueError(
                            f"Unmapped label value '{value}' in column '{label_from_column}'."
                        )
                return {"label": labels}

            dataset = dataset.map(
                map_label,
                batched=True,
                **_map_kwargs(map_num_proc, map_batch_size),
            )
            return dataset
        if not require_label:
            dataset = dataset.add_column("label", [-1] * len(dataset))
            return dataset
        raise ValueError(f"Label column '{label_column}' not found in dataset.")

    if label_column != "label":
        dataset = dataset.rename_column(label_column, "label")

    if dataset.features["label"].dtype == "string":
        dataset = dataset.class_encode_column("label")

    return dataset


def _add_index_column(
    dataset: Any,
    map_num_proc: Optional[int] = None,
    map_batch_size: Optional[int] = None,
) -> Any:
    def add_index(_, indices):
        return {"idx": indices}

    return dataset.map(
        add_index,
        with_indices=True,
        batched=True,
        **_map_kwargs(map_num_proc, map_batch_size),
    )


def _tokenize_dataset(
    dataset: Any,
    tokenizer,
    text_column: str,
    max_length: int,
    map_num_proc: Optional[int] = None,
    map_batch_size: Optional[int] = None,
) -> Any:
    if text_column not in dataset.column_names:
        raise ValueError(f"Text column '{text_column}' not found in dataset.")

    def tokenize_batch(batch):
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(
        tokenize_batch,
        batched=True,
        **_map_kwargs(map_num_proc, map_batch_size),
    )
    keep_columns = {"input_ids", "attention_mask", "label", "idx"}
    if "token_type_ids" in dataset.column_names:
        keep_columns.add("token_type_ids")
    remove_columns = [c for c in dataset.column_names if c not in keep_columns]
    dataset = dataset.remove_columns(remove_columns)
    dataset.set_format(type="torch")
    return dataset


def _limit_dataset(dataset: Any, max_rows: Optional[int]) -> Any:
    if max_rows is None:
        return dataset
    if len(dataset) <= max_rows:
        return dataset
    return dataset.select(range(max_rows))


def load_raid_bench(
    cfg: RaidBenchConfig,
    tokenizer: Optional[AutoTokenizer] = None,
) -> Tuple[Any, Optional[Any], Optional[Any]]:
    try:
        datasets = importlib.import_module("datasets")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install it with "
            "`pip install datasets` (activate your venv first if applicable)."
        ) from exc

    train = val = test = None
    dataset_dict = None

    if cfg.name_or_path in {"raid-bench", "raid"}:
        train, val, test = _load_from_raid_package(datasets, cfg)
    else:
        try:
            dataset_dict = datasets.load_dataset(cfg.name_or_path)
        except Exception:
            train, val, test = _load_from_raid_package(datasets, cfg)

    if dataset_dict is not None:
        train = _get_split(dataset_dict, cfg.split_train)
        val = _get_split(dataset_dict, cfg.split_val)
        test = _get_split(dataset_dict, cfg.split_test) if cfg.load_test else None

    if train is None:
        raise ValueError(f"Train split '{cfg.split_train}' not found.")

    train = _limit_dataset(train, cfg.max_rows)
    if cfg.val_from_train and val is None:
        split = train.train_test_split(
            test_size=float(cfg.val_fraction),
            seed=int(cfg.val_seed),
            shuffle=True,
        )
        train = split["train"]
        val = split["test"]

    if val is not None:
        val = _limit_dataset(val, cfg.max_rows)
    if test is not None:
        test = _limit_dataset(test, cfg.max_rows)

    tokenizer = tokenizer or AutoTokenizer.from_pretrained(cfg.name_or_path)

    train = _ensure_label_column(
        train,
        cfg.label_column,
        require_label=True,
        label_from_column=cfg.label_from_column,
        positive_values=cfg.positive_values,
        negative_values=cfg.negative_values,
        map_num_proc=cfg.map_num_proc,
        map_batch_size=cfg.map_batch_size,
    )
    train = _add_index_column(
        train,
        map_num_proc=cfg.map_num_proc,
        map_batch_size=cfg.map_batch_size,
    )
    train = _tokenize_dataset(
        train,
        tokenizer,
        cfg.text_column,
        cfg.max_length,
        map_num_proc=cfg.map_num_proc,
        map_batch_size=cfg.map_batch_size,
    )

    if val is not None:
        val = _ensure_label_column(
            val,
            cfg.label_column,
            require_label=True,
            label_from_column=cfg.label_from_column,
            positive_values=cfg.positive_values,
            negative_values=cfg.negative_values,
            map_num_proc=cfg.map_num_proc,
            map_batch_size=cfg.map_batch_size,
        )
        val = _add_index_column(
            val,
            map_num_proc=cfg.map_num_proc,
            map_batch_size=cfg.map_batch_size,
        )
        val = _tokenize_dataset(
            val,
            tokenizer,
            cfg.text_column,
            cfg.max_length,
            map_num_proc=cfg.map_num_proc,
            map_batch_size=cfg.map_batch_size,
        )

    if test is not None:
        test = _ensure_label_column(
            test,
            cfg.label_column,
            require_label=False,
            label_from_column=cfg.label_from_column,
            positive_values=cfg.positive_values,
            negative_values=cfg.negative_values,
            map_num_proc=cfg.map_num_proc,
            map_batch_size=cfg.map_batch_size,
        )
        test = _add_index_column(
            test,
            map_num_proc=cfg.map_num_proc,
            map_batch_size=cfg.map_batch_size,
        )
        test = _tokenize_dataset(
            test,
            tokenizer,
            cfg.text_column,
            cfg.max_length,
            map_num_proc=cfg.map_num_proc,
            map_batch_size=cfg.map_batch_size,
        )

    return train, val, test


def _raid_split_name(name: str) -> Optional[str]:
    name = name.lower()
    if name in {"train", "test", "extra"}:
        return name
    if name in {"validation", "val", "dev"}:
        return "test"
    return None


def _raid_filename(split: str, include_adversarial: bool) -> str:
    if include_adversarial:
        return f"{split}.csv"
    return f"{split}_none.csv"


def _load_from_raid_package(datasets, cfg: RaidBenchConfig):
    try:
        from raid.utils import RAID_CACHE_DIR, RAID_DATA_URL_BASE, download_file, load_data
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RAID package not found. Install it with `pip install raid-bench`."
        ) from exc

    train_split = _raid_split_name(cfg.split_train)
    val_split = _raid_split_name(cfg.split_val)
    test_split = _raid_split_name(cfg.split_test)

    if train_split is None:
        raise ValueError(f"Unsupported train split '{cfg.split_train}' for RAID.")

    if cfg.max_rows is not None:
        import pandas as pd

        def read_split(split_name: Optional[str]):
            if not split_name:
                return None
            fname = _raid_filename(split_name, cfg.include_adversarial)
            fp = download_file(f"{RAID_DATA_URL_BASE}/{fname}", RAID_CACHE_DIR / fname)
            return pd.read_csv(fp, nrows=int(cfg.max_rows))

        train_df = read_split(train_split)
        val_df = read_split(val_split)
        test_df = read_split(test_split) if cfg.load_test else None
    else:
        train_df = load_data(split=train_split, include_adversarial=cfg.include_adversarial)
        val_df = (
            load_data(split=val_split, include_adversarial=cfg.include_adversarial)
            if val_split
            else None
        )
        test_df = (
            load_data(split=test_split, include_adversarial=cfg.include_adversarial)
            if (test_split and cfg.load_test)
            else None
        )

    train = datasets.Dataset.from_pandas(train_df, preserve_index=False)
    val = datasets.Dataset.from_pandas(val_df, preserve_index=False) if val_df is not None else None
    test = datasets.Dataset.from_pandas(test_df, preserve_index=False) if test_df is not None else None

    return train, val, test
