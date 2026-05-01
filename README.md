# 🔍 Fake News Detector

An AI-powered fake news detection system built with **SVM + TF-IDF**, capable of classifying news articles as **FAKE** or **REAL** with confidence scores.

---

## 🚀 Live Demo
> Deployed on AWS EC2 — accessible 24/7  
> Dashboard: `http://your-ec2-ip:8501`  
> API: `http://your-ec2-ip:8080`

---

## 📌 Project Overview

This project detects fake news articles using Machine Learning. It fetches live news every hour via NewsAPI, classifies them using a trained SVM model, and displays results on an interactive Streamlit dashboard.

---

## ✅ Features

- 🤖 **SVM + TF-IDF model** trained on 55,000+ articles
- 📊 **Streamlit dashboard** with real-time predictions
- ⚡ **FastAPI backend** for fast REST API predictions
- ⏰ **Hourly scheduler** that fetches and classifies live news
- 🔄 **Auto-retraining** every 500 new articles
- 📈 **Dataset statistics** shown on dashboard
- 🌍 **Diverse training data** — politics, health, science, economy

---

## 🗂️ Project Structure

```
fake-news-detector/
│
├── explore.py          # Dataset preparation and merging
├── train_model.py      # SVM model training
├── predict.py          # Prediction script
├── api.py              # FastAPI backend (port 8080)
├── dashboard.py        # Streamlit dashboard (port 8501)
├── scheduler.py        # Hourly news fetcher + auto-retrainer
├── requirements.txt    # Python dependencies
│
├── data/               # Datasets (not pushed to GitHub)
│   ├── dataset.csv     # Merged training dataset
│   └── liar_dataset/   # LIAR dataset files
│
└── models/             # Trained model files (not pushed to GitHub)
    ├── svm_model.pkl
    └── tfidf_vectorizer.pkl
```

---

## 🧠 Model Details

| Component | Details |
|---|---|
| Algorithm | LinearSVC (Support Vector Machine) |
| Features | TF-IDF (unigrams + bigrams + trigrams) |
| Vocabulary | 150,000 features |
| Training data | 55,542 articles |
| Calibration | CalibratedClassifierCV |
| Class balance | class_weight="balanced" |

### 📊 Dataset Sources

| Dataset | Articles | Topics | Years |
|---|---|---|---|
| ISOT | ~44,000 | Politics | 2016–2017 |
| WELFake | ~72,000 | Mixed | 2015–2018 |
| LIAR | ~12,800 | Politics, Health, Science, Economy | 2007–2017 |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/Yashwantheswar/Fake-News-detector.git
cd Fake-News-detector
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
- **ISOT**: [Download True.csv and Fake.csv](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **LIAR**: [Download liar_dataset.zip](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip) → extract to `data/liar_dataset/`

### 5. Prepare dataset
```bash
python explore.py
```

### 6. Train the model
```bash
python train_model.py
```

---

## 🏃 Running the Project

Open **3 terminals** and run:

**Terminal 1 — API:**
```bash
uvicorn api:app --host 0.0.0.0 --port 8080
```

**Terminal 2 — Dashboard:**
```bash
streamlit run dashboard.py
```

**Terminal 3 — Hourly Scheduler:**
```bash
set NEWS_API_KEY=your_newsapi_key   # Windows
export NEWS_API_KEY=your_newsapi_key # Linux
python scheduler.py
```

Then open: `http://localhost:8501`

---

## 🌐 API Usage

### Predict single article
```bash
curl -X POST "http://localhost:8080/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Your news article text here"}'
```

### Response
```json
{
  "prediction": "FAKE",
  "confidence": 94.5,
  "message": "Likely fake news!"
}
```

### Check API health
```bash
curl http://localhost:8080/health
```

---

## 📈 How Hourly Update Works

```
Every hour
    ↓
Fetch 50 new articles from NewsAPI
    ↓
Classify each article with SVM model
    ↓
Save to dataset.csv
    ↓
Every 500 new articles → auto retrain
    ↓
Model improves continuously
```

---

## ☁️ AWS Deployment

Project is deployed on AWS EC2 (t2.micro - Free Tier):

```bash
# Connect to EC2
ssh -i fakenews-key.pem ubuntu@your-ec2-ip

# Run services using screen
screen -S api
uvicorn api:app --host 0.0.0.0 --port 8080

screen -S dashboard  
streamlit run dashboard.py --server.port 8501

screen -S scheduler
python scheduler.py
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| scikit-learn | SVM model |
| FastAPI | REST API backend |
| Streamlit | Web dashboard |
| Pandas / NumPy | Data processing |
| NewsAPI | Live news fetching |
| AWS EC2 | Cloud deployment |
| GitHub | Version control |

---

## 📋 Requirements

```
scikit-learn
fastapi
uvicorn
streamlit
pandas
numpy
joblib
requests
schedule
newsapi-python
```

---

## 👤 Author

**Yashwantheswar**  
GitHub: [@Yashwantheswar](https://github.com/Yashwantheswar)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
