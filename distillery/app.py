import datasets
import gradio as gr
import transformers
from distillery.distillator import distillate as distillate_


OUTPUT_DIR = "tmp/optimized_model"


def distillate(model_name, dataset_name, progress=gr.Progress()):
    progress(0, desc="Loading model")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)

    progress(0.05, desc="Loading dataset")
    dataset = datasets.load_dataset(dataset_name)

    distilled_model = distillate_(model, tokenizer, dataset, progress)

    distilled_model.model.save_pretrained(OUTPUT_DIR)

    return distilled_model.metrics


demo = gr.Interface(
    fn=distillate,
    inputs=[
        gr.Dropdown(["bert-base-cased"], label="Model", info="Choose a base model to optimize"),
        gr.Dropdown(["squad"], label="Dataset", info="Choose a dataset to use for training and evaluation")
    ],
    outputs=gr.Dataframe(headers=["", "accuracy", "latency", "size"], row_count=3, label="Result"),
    title="Distillery AI",
    description="A simple tool to optimize your model.",
)

demo.queue(concurrency_count=10).launch()
