import datasets
import gradio as gr
import transformers
from datasets import Dataset

from distillery.distillator import distillate as distillate_


OUTPUT_DIR = "tmp/optimized_model"


def distillate(model_name, dataset_name, target_gpu, sample, progress=gr.Progress()):
    progress(0, desc="Loading model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)

    progress(0.05, desc="Loading dataset")
    dataset = datasets.load_dataset(dataset_name)
    train_dataset: Dataset = dataset["train"]
    val_dataset: Dataset = dataset["validation"]

    if sample:
        train_dataset = train_dataset.select(range(100))
        val_dataset = val_dataset.select(range(20))

    distilled_model = distillate_(model, tokenizer, train_dataset, val_dataset, progress)

    # fixme: currently saving fails for sparsed models
    # distilled_model.model.save_pretrained(OUTPUT_DIR)

    return distilled_model.metrics


demo = gr.Interface(
    fn=distillate,
    inputs=[
        gr.Dropdown(["bert-base-cased"], label="Model", info="Choose a base model to optimize"),
        gr.Dropdown(["squad"], label="Dataset", info="Choose a dataset to use for training and evaluation"),
        gr.Dropdown(["RTX 4090", "A100", "H100"], label="Target GPU", info="Choose target GPU to optimize for"),
        gr.Checkbox(label="Sample data", info="Select to sample your train and validation datasets")
    ],
    outputs=gr.Dataframe(headers=["", "accuracy, %", "latency, s", "size"], row_count=3, label="Result"),
    title="Distillery AI",
    description="A simple tool to optimize your model.",
)

demo.queue(concurrency_count=10).launch()
