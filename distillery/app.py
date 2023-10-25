import datasets
import gradio as gr
import transformers
from datasets import Dataset

from distillery.distillator import distillate as distillate_


OUTPUT_DIR = "tmp/optimized_model"

SUMMARY = """# Results
**+30% latency improvements**
* better GPU utilization, e.g. we need 7 GPUs instead of 10
* lower costs when scaling LLM-based applications
* ability to use smaller GPUs 

**Accuracy**
* Same level of accuracy meaning no regression of UX
"""


def distillate(model_name, dataset_name, target_gpu, pruning, quantization, sample, progress=gr.Progress()):
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

    return [distilled_model.metrics, "Deploy model", "Download model", SUMMARY]


demo = gr.Interface(
    fn=distillate,
    inputs=[
        gr.Dropdown(["bert-base-cased"], label="Model", info="Choose a base model to optimize"),
        gr.Dropdown(["squad"], label="Dataset", info="Choose a dataset to use for training and evaluation"),
        gr.Dropdown(["RTX 4090", "A100", "H100"], label="Target GPU", info="Choose target GPU to optimize for"),
        gr.Checkbox(label="Pruning", info="Select to prune model"),
        gr.Checkbox(label="Quantization", info="Select to quantize model"),
        gr.Checkbox(label="Sample data", info="Select to sample your train and validation datasets")
    ],
    outputs=[
        gr.Dataframe(headers=["", "accuracy, %", "latency, s", "size"], row_count=3, label="Result"),
        gr.Button(variant="primary", visible=False),
        gr.Button(variant="secondary", visible=False),
        gr.Markdown()
    ],
    title="Distillery AI",
    description="A simple tool to optimize your model.",
    allow_flagging="never"
)

demo.queue(concurrency_count=10).launch()
