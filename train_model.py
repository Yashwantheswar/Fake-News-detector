import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import re

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH = "data/dataset.csv"
TEXT_COLUMN  = "content"
LABEL_COLUMN = "label"
MODEL_DIR    = "models"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)
df = df[[TEXT_COLUMN, LABEL_COLUMN]].dropna()

print(f"Total articles: {len(df)}")
print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts()}\n")

# Balance dataset
fake = df[df[LABEL_COLUMN] == 0]
real = df[df[LABEL_COLUMN] == 1]
min_count = min(len(fake), len(real), 25000)

fake = fake.sample(min_count, random_state=RANDOM_STATE)
real = real.sample(min_count, random_state=RANDOM_STATE)
df = pd.concat([fake, real]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

print(f"Training on {len(df)} articles ({min_count} fake, {min_count} real)\n")

# ─────────────────────────────────────────────
# 2. IMPROVED TEXT CLEANER
# ─────────────────────────────────────────────
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)         # remove URLs
    text = re.sub(r"\S+@\S+", "", text)                # remove emails
    text = re.sub(r"\d+", " ", text)                   # remove numbers
    text = re.sub(r"[^a-z\s]", "", text)               # keep only letters
    text = re.sub(r"\b\w{1,2}\b", "", text)            # remove 1-2 char words
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("Cleaning text...")
df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)

# Remove very short articles (less than 20 words) — noisy data
df = df[df["clean_text"].str.split().str.len() >= 20].reset_index(drop=True)
print(f"After removing short articles: {len(df)} rows\n")

# ─────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df[LABEL_COLUMN],
    test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[LABEL_COLUMN]
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

# ─────────────────────────────────────────────
# 4. IMPROVED TF-IDF
# ─────────────────────────────────────────────
print("Fitting TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(
    max_features=150_000,      # more features (was 100k)
    ngram_range=(1, 3),        # trigrams added (was 1,2)
    sublinear_tf=True,
    min_df=2,
    max_df=0.95,               # ignore words in 95%+ of docs
    analyzer="word",
    strip_accents="unicode",
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)
print(f"Vocabulary size: {len(vectorizer.vocabulary_)}\n")

# ─────────────────────────────────────────────
# 5. TRAIN SVM WITH CALIBRATION
# CalibratedClassifierCV gives real probabilities
# so confidence % is accurate (not raw decision scores)
# ─────────────────────────────────────────────
print("Training SVM (LinearSVC + Calibration)...")
base_model = LinearSVC(C=0.5, max_iter=3000, random_state=RANDOM_STATE,class_weight="balanced")
model = CalibratedClassifierCV(base_model, cv=3)
model.fit(X_train_tfidf, y_train)
print("Training complete!\n")

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
y_pred   = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
joblib.dump(model,      os.path.join(MODEL_DIR, "svm_model.pkl"))
joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))

with open(os.path.join(MODEL_DIR, "train_log.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Train size: {len(X_train)}\n")
    f.write(f"Test size: {len(X_test)}\n")

print(f"\nModel saved      → {MODEL_DIR}/svm_model.pkl")
print(f"Vectorizer saved → {MODEL_DIR}/tfidf_vectorizer.pkl")
print("\nDone! Now run: python predict.py")
