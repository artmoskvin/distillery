from dataclasses import dataclass
from typing import Any

import torch
import transformers
from datasets import Dataset
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured

from distillery.data import preprocess_train_function, preprocess_validation_function
from distillery.metrics import compute_metrics, measure_execution_time, measure_model_size


def distillate(model, tokenizer, train_dataset: Dataset, val_dataset: Dataset, progress_tracker=None):
    # Step 1. Set up train and val dataset
    if progress_tracker is not None:
        progress_tracker(0.1, desc="Preprocessing data")

    tokenized_dataset = {
        "train":
            train_dataset.map(
                lambda x: preprocess_train_function(x, tokenizer), batched=True
            ),
        "validation":
            val_dataset.map(
                lambda x: preprocess_validation_function(x, tokenizer),
                batched=True,
                remove_columns=train_dataset.column_names,
            )
    }

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 2. Train baseline model

    if progress_tracker is not None:
        progress_tracker(0.2, desc="Training baseline model (this can take a while)")

    training_args = transformers.TrainingArguments(
        "trainer",
        num_train_epochs=1,
        lr_scheduler_type="constant",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=256,
    )

    trainer = transformers.Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 2:4 sparsity require fp16, so we cast here for a fair comparison
    with torch.autocast("cuda"):
        with torch.inference_mode():
            predictions = trainer.predict(tokenized_dataset["validation"])

        baseline_metrics = get_metrics(val_dataset, model, predictions, tokenized_dataset)

    print("Baseline accuracy", baseline_metrics.accuracy)
    print("Baseline throughput", baseline_metrics.throughput)

    # Step 3. Prune model to be 2:4 sparse
    if progress_tracker is not None:
        progress_tracker(0.6, desc="Optimizing model (this can take a while)")

    sparsifier = WeightNormSparsifier(
        # apply sparsity to all blocks
        sparsity_level=1.0,
        # shape of 4 elemens is a block
        sparse_block_shape=(1, 4),
        # two zeros for every block of 4
        zeros_per_block=2
    )

    # add to config if nn.Linear and in the BERT model.
    sparse_config = [
        {"tensor_fqn": f"{fqn}.weight"}
        for fqn, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and "layer" in fqn
    ]

    # Prepare the model, insert fake-sparsity parameterizations for training
    sparsifier.prepare(model, sparse_config)
    sparsifier.step()

    trainer.train()

    sparsifier.squash_mask()

    model = model.cuda().half()
    # accelerate for sparsity
    for fqn, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "layer" in fqn:
            module.weight = torch.nn.Parameter(to_sparse_semi_structured(module.weight))

    with torch.inference_mode():
        predictions = trainer.predict(tokenized_dataset["validation"])

    optimized_metrics = get_metrics(val_dataset, model, predictions, tokenized_dataset)

    print("Summary:")
    print(f"Baseline accuracy: {baseline_metrics.accuracy}")
    print(f"New accuracy: {optimized_metrics.accuracy}")
    print(f"Baseline throughput: {baseline_metrics.throughput}")
    print(f"New throughput: {optimized_metrics.throughput}")
    print(f"Baseline model size: {baseline_metrics.memory} MB")
    print(f"New model size: {optimized_metrics.memory} MB")

    batch_size_with_max_diff = find_max_difference_key(baseline_metrics.throughput, optimized_metrics.throughput)

    metrics = [
        [
            "base model",
            round(baseline_metrics.accuracy["f1"], 2),
            round(baseline_metrics.throughput[batch_size_with_max_diff] / batch_size_with_max_diff, 2),
            round(baseline_metrics.memory, 2)
        ],
        [
            "optimized model",
            round(optimized_metrics.accuracy["f1"], 2),
            round(optimized_metrics.throughput[batch_size_with_max_diff] / batch_size_with_max_diff, 2),
            round(optimized_metrics.memory, 2)
        ],
        [
            "delta",
            round(optimized_metrics.accuracy['f1'] - baseline_metrics.accuracy['f1'], 2),
            f"{round(baseline_metrics.throughput[batch_size_with_max_diff] / optimized_metrics.throughput[batch_size_with_max_diff], 2)}x",
            f"{round(baseline_metrics.memory / optimized_metrics.memory, 2)}x"
        ]
    ]

    return DistillationResult(model, metrics)


@dataclass
class DistillationResult:
    model: Any
    metrics: list


def find_max_difference_key(dict1, dict2):
    max_diff = 0.0
    max_key = None

    for key in dict1.keys():
        diff = abs(dict1[key] - dict2[key])

        if diff > max_diff:
            max_diff = diff
            max_key = key

    return max_key


@dataclass
class Metric:
    accuracy: dict
    throughput: dict
    memory: float


def get_metrics(val_dataset, model, predictions, tokenized_dataset):
    # batch sizes to compare for eval
    batch_sizes = [4, 16, 64, 256]
    start_logits, end_logits = predictions.predictions

    accuracy = compute_metrics(
        start_logits,
        end_logits,
        tokenized_dataset["validation"],
        val_dataset,
    )

    throughput = measure_execution_time(
        model,
        batch_sizes,
        tokenized_dataset["validation"],
    )

    size = measure_model_size(model)

    return Metric(accuracy, throughput, size)
