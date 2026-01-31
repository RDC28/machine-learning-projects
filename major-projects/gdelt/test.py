import pandas as pd
import requests
import zipfile
import io
import time
import numpy as np
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import hdbscan

# ==================================================
# CONFIG (tune here, not everywhere)
# ==================================================
MAX_GDELT_ROWS = 5000        # safe cap
MAX_ARTICLES = 80            # headlines to fetch
REQUEST_TIMEOUT = 5
HEADLINE_DELAY = 0.25        # polite scraping delay
MIN_CLUSTER_SIZE = 5


# ==================================================
# 1. FAST GDELT FETCH (lastupdate.txt)
# ==================================================
def fetch_gdelt_latest():
    lastupdate_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
    text = requests.get(lastupdate_url, timeout=10).text

    mentions_url = None
    for line in text.splitlines():
        if "mentions" in line:
            mentions_url = line.split()[-1]
            break

    if not mentions_url:
        raise RuntimeError("No mentions file found in lastupdate.txt")

    print(f"[INFO] Fetching: {mentions_url}")

    zip_bytes = requests.get(mentions_url, timeout=15).content

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                encoding="latin-1",
                low_memory=False
            )

    print(f"[INFO] Loaded {len(df)} GDELT rows")
    return df


# ==================================================
# 2. HEADLINE EXTRACTION (LIGHT + SAFE)
# ==================================================
def fetch_article_title(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        soup = BeautifulSoup(r.text, "lxml")

        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            return og["content"].strip()

        if soup.title and soup.title.text:
            return soup.title.text.strip()

    except Exception:
        return None

    return None


def extract_real_headlines(df):
    urls = df[5].dropna().unique()[:MAX_ARTICLES]

    headlines = []

    print("[INFO] Fetching article headlines...")

    for url in urls:
        title = fetch_article_title(url)
        if title and len(title.split()) > 4:
            headlines.append(title)
        time.sleep(HEADLINE_DELAY)

    print(f"[INFO] Collected {len(headlines)} clean headlines")
    return headlines


# ==================================================
# 3. EMBEDDINGS (LOAD MODEL ONCE)
# ==================================================
def embed_texts(texts, model):
    print("[INFO] Creating embeddings...")
    return model.encode(texts, show_progress_bar=True)


# ==================================================
# 4. UNSUPERVISED CLUSTERING
# ==================================================
def cluster_embeddings(embeddings):
    print("[INFO] Clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE,
        metric="euclidean"
    )
    return clusterer.fit_predict(embeddings)


# ==================================================
# 5. PRINT DAILY HIGHLIGHTS
# ==================================================
def print_highlights(texts, embeddings, labels):
    clusters = {}

    for text, emb, label in zip(texts, embeddings, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append((text, emb))

    print(f"\n📰 FOUND {len(clusters)} NEWS CLUSTERS:\n")

    for label, items in clusters.items():
        texts_, embs_ = zip(*items)
        center = np.mean(embs_, axis=0)
        sims = np.dot(embs_, center)
        highlight = texts_[np.argmax(sims)]

        print(f"🟦 CLUSTER {label} ({len(items)} articles)")
        print(f"→ {highlight}")
        print("-" * 70)


# ==================================================
# 6. MAIN PIPELINE (FAST + SAFE)
# ==================================================
def run_pipeline():
    df = fetch_gdelt_latest()

    # SAFE SAMPLING (no crash ever)
    sample_size = min(MAX_GDELT_ROWS, len(df))
    df = df.sample(sample_size, random_state=42)

    texts = extract_real_headlines(df)

    if len(texts) < 10:
        print("[WARN] Not enough articles to cluster.")
        return

    # Load embedding model once
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = embed_texts(texts, model)
    labels = cluster_embeddings(embeddings)
    print_highlights(texts, embeddings, labels)


# ==================================================
# ENTRY POINT
# ==================================================
if __name__ == "__main__":
    run_pipeline()

