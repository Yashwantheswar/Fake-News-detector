import joblib
import re
import os

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_DIR = "models"

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model_path = os.path.join(MODEL_DIR, "svm_model.pkl")
vec_path   = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

if not os.path.exists(model_path) or not os.path.exists(vec_path):
    raise FileNotFoundError(
        "Model not found. Run train_model.py first!\n"
        f"Expected: {model_path} and {vec_path}"
    )

_model      = joblib.load(model_path)
_vectorizer = joblib.load(vec_path)

# ─────────────────────────────────────────────
# TEXT CLEANER — must match train_model.py
# ─────────────────────────────────────────────
def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\b\w{1,2}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ─────────────────────────────────────────────
# PREDICT FUNCTION
# Same signature as original: predict(text) → label, confidence
# api.py and scheduler.py work with zero changes
# ─────────────────────────────────────────────
def predict(text: str):
    """
    Returns:
        label      (str)  : "FAKE" or "REAL"
        confidence (float): 0.0 – 100.0
    """
    cleaned   = _clean_text(text)
    tfidf_vec = _vectorizer.transform([cleaned])

    # CalibratedClassifierCV gives real probability — much more accurate
    proba      = _model.predict_proba(tfidf_vec)[0]   # [prob_fake, prob_real]
    raw_label  = _model.predict(tfidf_vec)[0]

    # Confidence = probability of the predicted class
    confidence = round(float(max(proba)) * 100, 2)
    label      = "FAKE" if int(raw_label) == 0 else "REAL"

    return label, confidence


# ─────────────────────────────────────────────
# CLI TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Fake News Detector ready!")
    print("Tip: paste a FULL article or paragraph, not just a headline")
    print("Type 'quit' to exit\n")

    while True:
        text = input("Paste a news article: ")
        if text.lower() == "quit":
            break
        label, conf = predict(text)
        print(f"Prediction: {label} ({conf}% confidence)\n")
