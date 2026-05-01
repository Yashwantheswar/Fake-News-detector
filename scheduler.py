from apscheduler.schedulers.blocking import BlockingScheduler
from newsapi import NewsApiClient
from predict import predict
import pandas as pd, os, hashlib
from dotenv import load_dotenv

load_dotenv()
api = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
DATASET_PATH = "data/dataset.csv"
SEEN_PATH = "data/seen_hashes.txt"

def get_seen():
    if not os.path.exists(SEEN_PATH):
        return set()
    return set(open(SEEN_PATH).read().splitlines())

def save_seen(seen):
    open(SEEN_PATH, "w").write("\n".join(seen))

def fetch_and_update():
    print("Fetching new articles...")
    seen = get_seen()
    articles = api.get_everything(
        q="politics OR science OR health OR technology",
        language="en",
        page_size=50
    )["articles"]

    new_rows = []
    for a in articles:
        text = f"{a['title'] or ''} {a['description'] or ''}"
        if len(text.strip()) < 20:
            continue
        h = hashlib.md5(text.encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        label_str, conf = predict(text)
        if conf < 75:
            print(f"Skipping low-confidence: {conf}%")
            continue
        label = 0 if label_str == "FAKE" else 1
        new_rows.append({"text": text, "label": label})

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_existing = pd.read_csv(DATASET_PATH)
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates()
        df_combined.to_csv(DATASET_PATH, index=False)
        print(f"Added {len(new_rows)} articles. Total: {len(df_combined)}")

        # Auto retrain every 500 new articles
        last_count = 0
        if os.path.exists("data/last_train_count.txt"):
            last_count = int(open("data/last_train_count.txt").read())
        if len(df_combined) - last_count >= 500:
            print("Retraining model with new data...")
            os.system("python train_model.py")
            open("data/last_train_count.txt", "w").write(str(len(df_combined)))
    else:
        print("No new articles found.")

    save_seen(seen)
    print("Done!\n")

# Run once immediately then every hour
print("Scheduler started — fetching articles every hour")
fetch_and_update()

scheduler = BlockingScheduler()
scheduler.add_job(fetch_and_update, "interval", hours=1)
scheduler.start()