import pandas as pd
import os

# ─────────────────────────────────────────────
# 1. LOAD EXISTING dataset.csv (already built)
# ─────────────────────────────────────────────
print("Loading existing dataset.csv...")
df_existing = pd.read_csv("data/dataset.csv")

# Make sure columns are correct
if "content" not in df_existing.columns:
    # try to find text column
    for col in ["text", "Text", "Content", "article"]:
        if col in df_existing.columns:
            df_existing = df_existing.rename(columns={col: "content"})
            break

df_existing = df_existing[["content", "label"]].dropna()
print(f"   Existing dataset: {len(df_existing)} articles")
print(f"   REAL: {(df_existing['label']==1).sum()} | FAKE: {(df_existing['label']==0).sum()}\n")

# ─────────────────────────────────────────────
# 2. LOAD LIAR DATASET (already downloaded)
# ─────────────────────────────────────────────
liar = pd.DataFrame()
liar_path = "data/liar_dataset/train.tsv"

if os.path.exists(liar_path):
    print("Loading LIAR dataset...")
    liar_raw = pd.read_csv(liar_path, sep="\t", header=None,
                           names=["id","label_str","statement","subject",
                                  "speaker","job","state","party",
                                  "barely_true","false","half_true",
                                  "mostly_true","pants_on_fire","context"])

    fake_labels = ["pants-fire", "false", "barely-true"]
    real_labels = ["mostly-true", "true", "half-true"]
    liar_raw = liar_raw[liar_raw["label_str"].isin(fake_labels + real_labels)]
    liar_raw["label"]   = liar_raw["label_str"].apply(lambda x: 0 if x in fake_labels else 1)
    liar_raw["content"] = liar_raw["statement"]
    liar = liar_raw[["content", "label"]].dropna()
    print(f"   LIAR dataset: {len(liar)} articles")
    print(f"   REAL: {(liar['label']==1).sum()} | FAKE: {(liar['label']==0).sum()}\n")
else:
    # try valid.tsv or test.tsv if train.tsv not found
    for alt in ["data/liar_dataset/valid.tsv", "data/liar_dataset/test.tsv"]:
        if os.path.exists(alt):
            print(f"   Found {alt} instead of train.tsv — using it")
            liar_raw = pd.read_csv(alt, sep="\t", header=None,
                                   names=["id","label_str","statement","subject",
                                          "speaker","job","state","party",
                                          "barely_true","false","half_true",
                                          "mostly_true","pants_on_fire","context"])
            fake_labels = ["pants-fire", "false", "barely-true"]
            real_labels = ["mostly-true", "true", "half-true"]
            liar_raw = liar_raw[liar_raw["label_str"].isin(fake_labels + real_labels)]
            liar_raw["label"]   = liar_raw["label_str"].apply(lambda x: 0 if x in fake_labels else 1)
            liar_raw["content"] = liar_raw["statement"]
            liar = liar_raw[["content", "label"]].dropna()
            print(f"   LIAR: {len(liar)} articles loaded\n")
            break

    if len(liar) == 0:
        print("   ⚠️  No LIAR .tsv file found inside data/liar_dataset/")
        print("   Run: dir data\\liar_dataset\\ to check file names\n")

# ─────────────────────────────────────────────
# 3. MERGE
# ─────────────────────────────────────────────
print("Merging datasets...")
all_dfs = [df_existing]
if len(liar) > 0:
    all_dfs.append(liar)

df = pd.concat(all_dfs).dropna().drop_duplicates(subset=["content"])
df = df[df["content"].str.split().str.len() >= 10]  # remove very short
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv("data/dataset.csv", index=False)

print(f"\n{'='*50}")
print(f"Total articles : {len(df)}")
print(f"REAL (label=1) : {(df['label']==1).sum()}")
print(f"FAKE (label=0) : {(df['label']==0).sum()}")
print(f"{'='*50}")
print("\nDataset saved → data/dataset.csv")
print("Now run: python train_model.py")