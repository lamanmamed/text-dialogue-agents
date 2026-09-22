"""Settings and evaluation helpers for Gemma conversational fine-tuning."""

from __future__ import annotations

import random
from collections import Counter
import re
import string

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    gold = normalize_answer(reference).split()
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    overlap = Counter(predicted) & Counter(gold)
    same = sum(overlap.values())
    if same == 0:
        return 0.0
    precision = same / len(predicted)
    recall = same / len(gold)
    return 2 * precision * recall / (precision + recall)


def full_sft_settings() -> dict:
    """Training recipe used for full Gemma-3-270M fine-tuning."""
    return {
        "max_length": 256,
        "packing": False,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 2,
        "per_device_eval_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "learning_rate": 2e-5,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "linear",
        "weight_decay": 0.01,
        "optim": "paged_adamw_8bit",
        "logging_steps": 5,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "dataset_kwargs": {"add_special_tokens": False},
    }


def lora_settings() -> tuple[dict, dict]:
    """Return the LoRA adapter settings and its SFT training settings."""
    adapter = {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": "all-linear",
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    training = full_sft_settings()
    training["learning_rate"] = 1e-4
    return adapter, training


def load_smoltalk_splits(seed: int = 42):
    """Load everyday-conversations and reproduce the train/eval split used in the experiment."""
    try:
        from datasets import DatasetDict, load_dataset
    except ImportError as exc:
        raise ImportError("Install the project with the 'experiments' extra") from exc
    dataset = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations")
    split = dataset["train"].train_test_split(test_size=0.05, seed=seed)
    return DatasetDict({"train": split["train"], "eval": split["test"], "test": dataset["test"]})


def build_lora_config():
    try:
        from peft import LoraConfig
    except ImportError as exc:
        raise ImportError("Install the project with the 'experiments' extra") from exc
    adapter, _ = lora_settings()
    return LoraConfig(**adapter)
