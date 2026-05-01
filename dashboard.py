import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detector")
st.markdown("Powered by DistilBERT — updates every hour with new articles")

# ── Prediction section ──────────────────────────────────────
st.subheader("🔍 Check an article")
text = st.text_area("Paste a news article or headline below", height=150)

if st.button("Analyse", type="primary"):
    if text.strip():
        with st.spinner("Analysing..."):
            try:
                res = requests.post(
                    "http://127.0.0.1:8080/predict",
                    json={"text": text}
                )
                data = res.json()
                col1, col2 = st.columns(2)
                with col1:
                    if data["prediction"] == "FAKE":
                        st.error(f"🚨 FAKE NEWS ({data['confidence']}% confidence)")
                    else:
                        st.success(f"✅ REAL NEWS ({data['confidence']}% confidence)")
                with col2:
                    st.progress(data["confidence"] / 100)
                    st.caption(data["message"])
            except:
                st.error("API not running. Start it with: uvicorn api:app --port 8080")
    else:
        st.warning("Please paste some text first!")

st.divider()

# ── Dataset stats section ───────────────────────────────────
st.subheader("📊 Dataset Statistics")
try:
    stats = requests.get("http://127.0.0.1:8080/stats").json()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", stats["total_articles"])
    col2.metric("Fake Articles", stats["fake_articles"])
    col3.metric("Real Articles", stats["real_articles"])
except:
    st.warning("Could not load stats.")

st.divider()

# ── Dataset preview ─────────────────────────────────────────
st.subheader("🗂 Recent Dataset Entries")
try:
    df = pd.read_csv("data/dataset.csv")
    df["type"] = df["label"].map({0: "FAKE", 1: "REAL"})
    st.dataframe(df[["text", "type"]].tail(20), use_container_width=True)
except:
    st.warning("Dataset not found.")