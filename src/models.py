import torch
from transformers import pipeline

# The default model is "distilbert-base-uncased-finetuned-sst-2-english"
model_1_pipeline = pipeline("sentiment-analysis")
def predict_model_1(text):
    result = model_1_pipeline(text[:512])[0]
    return result["label"].lower(), result["score"]


MODEL_2_NAME = "siebert/sentiment-roberta-large-english"

model_2 = pipeline(
    task="sentiment-analysis",
    model=MODEL_2_NAME,
)


def predict_model_2(text: str):
    result = model_2(text[:512])[0]
    return result["label"], float(result["score"])