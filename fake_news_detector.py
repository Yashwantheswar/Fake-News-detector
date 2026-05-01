import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import joblib
import os

# --- Step 1: Data Acquisition ---
print("Checking for datasets...")
if not os.path.exists('Fake.csv') or not os.path.exists('True.csv'):
    print("Error: Fake.csv or True.csv not found in the current folder!")
    exit()

print("Loading 44,898 articles...") [cite: 48]
fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

# --- Step 2: Preprocessing ---
print("Combining and preprocessing text...") [cite: 50, 52]
fake['label'] = 1
true['label'] = 0
df = pd.concat([fake, true], axis=0)
df['content'] = (df['title'] + " " + df['text']).str.lower()

# --- Step 3: Feature Extraction ---
print("Extracting 5000 word features via TF-IDF...") [cite: 56, 57]
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
X = tfidf.fit_transform(df['content'])
y = df['label']

# --- Step 4: Model Training ---
print("Training Linear SVM Model...") [cite: 61]
model = LinearSVC()
model.fit(X, y)

# --- Step 5: Model Persistence ---
print("Saving model for production pipeline...") [cite: 73]
joblib.dump(model, 'svm_model.pkl')
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')

print("\nSUCCESS: Model trained and saved!")