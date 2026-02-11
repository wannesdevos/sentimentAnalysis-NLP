from fastapi import FastAPI
from models import predict_model_1, predict_model_2
app = FastAPI()

@app.post("/sentiment-analysis/model-1")
def sentiment_analysis_model_1(text: str):
    label, score = predict_model_1(text)
    return {"label": label, "score": score}

@app.post("/sentiment-analysis/model-2")   
def sentiment_analysis_model_2(text: str):
    label, score = predict_model_2(text)
    return {"label": label, "score": score}

