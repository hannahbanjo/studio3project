import json, re
from pathlib import Path
from collections import Counter
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

JSON_PATH = Path("fraud_results.json")

TEXT_COL        = "cleaned_text"
CLASS_COL       = "fraud_related"
REASON_COL      = "fraud_reason"
CLUSTER_COL     = "kmeans_cluster"

EXTRA_STOPS = {
    "occ","fdic","frs","federal","reserve","treasury","office","department",
    "united","states","u","s","section","bank","banks","banking","institution",
    "institutions","agency","agencies","newsroom","pdf","page","pages","date",
    "bulletin","press","release","public","policy","regulation","regulatory",
    "comment","comments","docket","system","board","governors"
}

TREND_MAP = {
    "phish": "Phishing / Social engineering",
    "impersonat": "Identity theft / ATO",
    "identity": "Identity theft / ATO",
    "account takeover": "Identity theft / ATO",
    "check": "Check fraud",
    "peer": "P2P payment scams",
    "zelle": "P2P payment scams",
    "invest": "Investment / Crypto scams",
    "crypto": "Investment / Crypto scams",
    "wire": "Wire / Transfer scams",
    "ach": "ACH / Transfer fraud",
    "elder": "Elder financial exploitation",
    "scam": "General scams"
}

insurance_keywords = ["claim", "payout", "collision", "adjuster", "policyholder"]
banking_keywords = ["zelle", "wire", "account takeover", "credit card", "ACH"]
investing_keywords = ["retirement", "IRA", "rollover", "brokerage", "portfolio"]

def load_any_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        txt = f.read().strip()
    return json.loads(txt)

def extract_top_keywords(text, top_n=20):
    text = text.lower()
    tokens = re.findall(r"\b[a-z]{3,}\b", text)
    stops = set(stopwords.words("english")).union(EXTRA_STOPS)
    tokens = [t for t in tokens if t not in stops]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    counts = Counter(tokens)
    
    return [word for word, freq in counts.most_common(top_n)]

def assign_trend(reason_text: str) -> str:
    t = (reason_text or "").lower()
    for needle, label in TREND_MAP.items():
        if needle in t:
            return label
    return "Other / General fraud"

def detect_lob(text):
    txt = text.lower()
    scores = {
        "Insurance": sum(word in txt for word in insurance_keywords), 
        "Banking": sum(word in txt for word in banking_keywords), 
        "Investing": sum(word in txt for word in investing_keywords)}

    return max(scores, key=scores.get)

def extract_tokens(text):
    text = text.lower().strip()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    stopset = set(stopwords.words("english")).union(EXTRA_STOPS)
    tokens = [w for w in words if w not in stopset and not w.isdigit()]
    return tokens

def classify_trend(reason):
    r = reason.lower()
    for keyword, trend in TREND_MAP.items():
        if keyword in r:
            return trend
    return "Other / General fraud"

def main():
    records = load_any_json(JSON_PATH)
    df = pd.json_normalize(records, sep=".")

    if CLASS_COL in df.columns:
        df = df[df[CLASS_COL] == True]

    df["tokens"] = df[TEXT_COL].astype(str).apply(extract_tokens)
    all_tokens = [tok for row in df["tokens"] for tok in row]
    top20 = Counter(all_tokens).most_common(20)
    print("\nTop 20 keywords overall:", top20)

    # Top 20 keywords chart
    if top20:
        labels = [w for w,_ in top20]
        values = [c for _,c in top20]
        plt.figure()
        plt.bar(labels, values)
        plt.title("Top 20 Keywords (fraud-related articles)")
        plt.tight_layout()
        plt.savefig("top20_keywords.png", dpi=200)

    df["LOB"] = df[TEXT_COL].astype(str).apply(detect_lob)
    lob_counts = df["LOB"].value_counts()
    print("\nLine of Business counts:\n", lob_counts)

    lob_keywords =  {}
    for lob, rows in df.groupby("LOB"):
        tokens = [tok for row in rows["tokens"] for tok in row]
        top_tokens = Counter(tokens)
        lob_keywords[lob] = top_tokens.most_common(10)
    print("\nTop keywords by Line of Business:")
    for lob, keywords in lob_keywords.items():
        print(f" {lob}: {keywords}")


    # Trends Chart
    if REASON_COL in df.columns:
        df["trend"] = df[REASON_COL].astype(str).apply(classify_trend)
        trend_counts = df["trend"].value_counts()
        print("\nTrend counts:\n", trend_counts)

        # Chart: Trends
        plt.figure()
        trend_counts.plot(kind="bar")
        plt.title("Fraud Trends (from LLM reasons)")
        plt.ylabel("Articles")
        plt.tight_layout()
        plt.savefig("top_trends.png", dpi=200)
        print("Saved: top_trends.png")

if __name__ == "__main__":
    main()