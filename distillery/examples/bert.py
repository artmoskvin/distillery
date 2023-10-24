import datasets
import transformers

from distillery.distillator import distillate

model_name = "bert-base-cased"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model = transformers.AutoModelForQuestionAnswering.from_pretrained(model_name)
dataset = datasets.load_dataset("squad")

distilled_model = distillate(model, tokenizer, dataset)

