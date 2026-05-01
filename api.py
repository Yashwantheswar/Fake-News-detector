from fastapi import FastAPI
from pydantic import BaseModel
from predict import predict
import pandas as pd
import os

app = FastAPI(title="Fake News Detector API")

# Get the directory where api.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")

class Article(BaseModel):
    text: str

@app.post("/predict")
def detect(article: Article):
    label, confidence = predict(article.text)
    return {
        "prediction": label,
        "confidence": confidence,
        "message": "Likely fake news!" if label == "FAKE" else "Appears to be real news."
    }

@app.get("/stats")
def stats():
    try:
        df = pd.read_csv(DATASET_PATH, usecols=["label"])
        return {
            "total_articles": len(df),
            "fake_articles": int((df["label"] == 0).sum()),
            "real_articles": int((df["label"] == 1).sum()),
        }
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}