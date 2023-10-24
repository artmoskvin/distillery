from dataclasses import dataclass

import torch
import transformers
from torch.ao.pruning import WeightNormSparsifier
from torch.sparse import to_sparse_semi_structured

from distillery.data import preprocess_train_function, preprocess_validation_function
from distillery.metrics import compute_metrics, measure_execution_time


def distillate(model, tokenizer, dataset):
    # Step 1. Set up train and val dataset

    tokenized_dataset = {
        "train":
            dataset["train"].map(
                lambda x: preprocess_train_function(x, tokenizer), batched=True
            ),
        "validation":
            dataset["validation"].map(
                lambda x: preprocess_validation_function(x, tokenizer),
                batched=True,
                remove_columns=dataset["train"].column_names,
            )
    }

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    # Step 2. Train baseline model

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

        baseline_metrics = get_metrics(dataset, model, predictions, tokenized_dataset)

    print("Baseline accuracy", baseline_metrics.accuracy)
    print("Baseline throughput", baseline_metrics.throughput)

    # Step 3. Prune model to be 2:4 sparse

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

    optimized_metrics = get_metrics(dataset, model, predictions, tokenized_dataset)

    print("Summary:")
    print(f"Baseline accuracy: {baseline_metrics.accuracy}")
    print(f"New accuracy: {optimized_metrics.accuracy}")
    print(f"Baseline throughput: {baseline_metrics.throughput}")
    print(f"New throughput: {optimized_metrics.throughput}")

    return model


@dataclass
class Metric:
    accuracy: dict
    throughput: dict
    memory: str


def get_metrics(dataset, model, predictions, tokenized_dataset):
    # batch sizes to compare for eval
    batch_sizes = [4, 16, 64, 256]
    start_logits, end_logits = predictions.predictions

    accuracy = compute_metrics(
        start_logits,
        end_logits,
        tokenized_dataset["validation"],
        dataset["validation"],
    )

    throughput = measure_execution_time(
        model,
        batch_sizes,
        tokenized_dataset["validation"],
    )

    return Metric(accuracy, throughput, "")
