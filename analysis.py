import json, re
from pathlib import Path
from collections import Counter
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.decomposition import PCA
import numpy as np

# ---------- 
# USAA THEME
# ----------
USAA_NAVY = "#002F6C"
USAA_GOLD = "#CC9900"
USAA_BABY_BLUE = "#A7C7E7"
USAA_SLATE = "#4D4D4F"
USAA_LIGHT_GRAY = "#A7A8AA"

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": USAA_SLATE,
    "axes.labelcolor": USAA_SLATE,
    "xtick.color": USAA_SLATE,
    "ytick.color": USAA_SLATE,
    "grid.color": USAA_LIGHT_GRAY,
    "grid.linestyle": "--",
    "grid.linewidth": 0.7,
    "axes.grid": False,        
    "font.size": 12,
    "axes.titleweight": "bold",
})

USAA_CMAP = LinearSegmentedColormap.from_list(
    "usaa_cmap", [USAA_NAVY, USAA_BABY_BLUE, "white", USAA_GOLD]
)

BOLD_USAA_CMAP = LinearSegmentedColormap.from_list(
    "usaa_cmap", [USAA_NAVY, USAA_GOLD]
)

# ----------
# Setup
# ---------- 
 
JSON_PATH   = Path("fraud_results_final.json")
TEXT_COL    = "cleaned_text"
CLASS_COL   = "fraud_related"
REASON_COL  = "fraud_reason"
DATE_COL    = "date"

EXTRA_STOPS = {
    "http","https","www","com","org","gov","edu","php","html","amp", "office currency" 
}

# ---------- 
# LOB Keywords
#  ----------
INSURANCE_KEYWORDS = [
    "claim", "claims", "payout", "collision", "adjuster", "policyholder", 
    "coverage", "premium", "underwriting", "auto insurance", "home insurance"
]
BANKING_KEYWORDS = [
    "zelle", "wire", "account takeover", "credit card", "ach", "atm", 
    "debit card", "fraudulent transaction", "overdraft", "online banking"
]
INVESTING_KEYWORDS = [
    "retirement", "ira", "rollover", "brokerage", "portfolio", 
    "stocks", "bonds", "mutual fund", "investment scam", "crypto", "bitcoin"
]

# ---------- 
# Fraud Trend Mapping
#  ----------
TREND_MAP = {
    r"phish": "Phishing / Social Engineering",
    r"vish": "Phishing / Social Engineering",
    r"impersonat": "Identity Theft / Account Takeover",
    r"identity": "Identity Theft / Account Takeover",
    r"account takeover": "Identity Theft / Account Takeover",
    r"credential stuffing": "Credential Attacks",
    r"sim swap": "SIM Swap Fraud",
    r"check": "Check Fraud",
    r"peer to peer|p2p": "P2P Payment Scams",
    r"zelle": "P2P Payment Scams",
    r"venmo": "P2P Payment Scams",
    r"wire": "Wire / Transfer Scams",
    r"ach": "ACH / Transfer Fraud",
    r"invest": "Investment / Crypto Scams",
    r"crypto": "Investment / Crypto Scams",
    r"bitcoin": "Investment / Crypto Scams",
    r"nft": "NFT / Digital Asset Scams",
    r"elder": "Elder Financial Exploitation",
}

def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.loads(f.read().strip())

def clean_text_strong(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    junk_phrases = ["skip to main content","main content","skip navigation","privacy policy",
                    "terms of service","terms of use","cookie policy","footer","header",
                    "home menu","navigation menu","menu","sidebar","log in","login",
                    "sign in","read more","click here","copyright","all rights reserved",
                    "proposed rule", "financial institution", "comptroller", "fdic",
                    "united states", "federal reserve", "secrecy act", "bank secrecy", "office",
                    "helpwithmybank", "banks", "search", "gov", "bank", "organization", "currency"
                    "management", "office currency", "custody services", "occ", 
                    "reputation", "risk management"]
    for jp in junk_phrases: text = text.replace(jp, " ")
    words = [w for w in text.split() if w not in EXTRA_STOPS]
    return " ".join(words)

def extract_ngrams(text, n=1):
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    words = [w for w in words if w not in ENGLISH_STOP_WORDS and len(w)>2]
    if n == 1:
        return Counter(words)
    return Counter([" ".join(words[i:i+n]) for i in range(len(words)-n+1)])

def assign_trend(reason_text: str) -> str:
    t = (reason_text or "").lower()
    for pattern, label in TREND_MAP.items():
        if re.search(pattern, t):
            return label
    return label

def detect_lob(text: str) -> str:
    txt = (text or "").lower()
    scores = {
        "Insurance": sum(word in txt for word in INSURANCE_KEYWORDS),
        "Banking": sum(word in txt for word in BANKING_KEYWORDS),
        "Investing": sum(word in txt for word in INVESTING_KEYWORDS),
    }
    return max(scores, key=scores.get)


def main():
    df = pd.json_normalize(load_json(JSON_PATH), sep=".")
    print(f"Loaded {len(df)} records")

    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df["year"] = df[DATE_COL].dt.year.astype("Int64")

    if CLASS_COL in df.columns:
        df = df[df[CLASS_COL]==True]
    if df.empty:
        print("No fraud-related articles found."); return

    df["cleaned_strong"] = df[TEXT_COL].astype(str).apply(clean_text_strong)
    combined_text = " ".join(df["cleaned_strong"])

    # Calculate Bigrams
    bigrams = extract_ngrams(combined_text, n=2)
    top_terms = bigrams.most_common(15)
    print("\nTop Bigrams:")
    for term, freq in top_terms:
        print(f"{term}: {freq}")

    # LOB Detection 
    df["LOB"] = df[TEXT_COL].astype(str).apply(detect_lob)
    print("\nLOB counts:\n", df["LOB"].value_counts())

    #  Trend Detection 
    if REASON_COL in df.columns:
        df["trend"] = df[REASON_COL].astype(str).apply(assign_trend)
        trend_counts = df["trend"].value_counts()
        print("\nTrend counts:\n", trend_counts)

    #  Plot top terms 
    plt.figure(figsize=(10,6))
    plt.bar([t[0] for t in top_terms], [t[1] for t in top_terms], color=USAA_NAVY, edgecolor=USAA_SLATE)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top Unigrams & Bigrams")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("top_terms.png", dpi=260)
    print("Saved: top_terms.png")

    #  Top trends by LOB 
    lob_trends = {}
    for lob, rows in df.groupby("LOB"):
        if REASON_COL in rows.columns:
            trends = rows[REASON_COL].astype(str).apply(assign_trend)
            counts = trends.value_counts().head(7)  # Top 7 trends per LOB
            lob_trends[lob] = counts

    lob_colors = {"Banking": USAA_NAVY, "Insurance": USAA_GOLD, "Investing": USAA_BABY_BLUE}
    all_trends, all_counts, all_colors = [], [], []

    for lob, counts in lob_trends.items():
        for trend, count in counts.items():
            all_trends.append(trend)
            all_counts.append(count)
            all_colors.append(lob_colors.get(lob, USAA_SLATE))

    plt.figure(figsize=(12,6))
    plt.bar(range(len(all_trends)), all_counts, color=all_colors, edgecolor=USAA_SLATE)
    plt.xticks(range(len(all_trends)), all_trends, rotation=45, ha="right")
    handles = [plt.Line2D([0],[0], color=c,lw=10,label=l) for l,c in lob_colors.items()]
    plt.legend(handles=handles, title="Line of Business", loc="upper right")
    plt.title("Top Fraud Trends by Line of Business")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("lob_trends.png", dpi=260)
    print("Saved: lob_trends.png")


    # Articles by Year 
    if "year" in df.columns and df["year"].notna().any():
        years = df["year"].dropna().astype(int)
        years_filtered = years[years != 2000]
        year_counts = years_filtered.value_counts().sort_index()
        plt.figure()
        plt.bar(year_counts.index.astype(str), year_counts.values, color=USAA_NAVY, edgecolor=USAA_SLATE)
        plt.title("Articles by Year")
        plt.ylabel("Number of Articles")
        plt.xlabel("Year")
        plt.tight_layout()
        plt.savefig("articles_by_year.png", dpi=260)
        print("Saved: articles_by_year.png")

    # K-Means Cluster Visualization
    if "embedding" in df.columns and "kmeans_cluster" in df.columns:
        print("\nPlotting K-Means Clusters...")
        
        embedding_data = np.array(df["embedding"].tolist())
        clusters = df["kmeans_cluster"].values
        
        pca = PCA(n_components=2)
        components = pca.fit_transform(embedding_data)
        
        plt.figure(figsize=(10, 8))
        
        cluster_colors = [USAA_NAVY, USAA_GOLD, USAA_BABY_BLUE, USAA_SLATE, USAA_LIGHT_GRAY][:len(np.unique(clusters))]
        
        for cluster_id in np.unique(clusters):
            idx = clusters == cluster_id
            
            plt.scatter(
                components[idx, 0], 
                components[idx, 1], 
                c=cluster_colors[cluster_id], 
                label=f"Cluster {cluster_id}",
                alpha=0.6,
                edgecolors=USAA_SLATE,
                linewidths=0.5
            )

        plt.title(f"K-Means Clusters (k={len(np.unique(clusters))}) visualized via PCA")
        plt.xlabel(f"PCA Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        plt.ylabel(f"PCA Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        plt.legend(title="Cluster ID")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("kmeans_clusters_pca.png", dpi=260)
        print("Saved: kmeans_clusters_pca.png")

    if "kmeans_cluster" in df.columns and "cleaned_strong" in df.columns:
        print("\nGenerating Word Clouds for each cluster...")

        try:
            from wordcloud import WordCloud
        except ImportError:
            print("ERROR: The 'wordcloud' library is not installed. Please run 'pip install wordcloud'.")
            return
        
        cluster_colors_visible = [USAA_NAVY, USAA_GOLD, USAA_SLATE]
        
        for cluster_num, color in zip(df["kmeans_cluster"].unique(), cluster_colors_visible):
            cluster_text = " ".join(df[df["kmeans_cluster"] == cluster_num]["cleaned_strong"].dropna())
            
            if not cluster_text:
                print(f"Skipping Cluster {cluster_num}: No substantial text found.")
                continue

            wordcloud = WordCloud(
                width=800, 
                height=400, 
                background_color="white",
                colormap=BOLD_USAA_CMAP, 
                stopwords=ENGLISH_STOP_WORDS 
            ).generate(cluster_text)
            
            filename = f"cluster_{cluster_num}_wordcloud.png"
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Cluster {cluster_num} Word Cloud", color=color) 
            plt.tight_layout(pad=0)
            plt.savefig(filename, dpi=260)
            print(f"Saved: {filename}")

    # Trend by Year Heatmap 
    if "year" in df.columns and REASON_COL in df.columns:
        valid_mask = df["year"].notna()
        table = pd.crosstab(df.loc[valid_mask,"year"], df.loc[valid_mask,"trend"])
        if not table.empty:
            plt.figure(figsize=(11,5))
            plt.imshow(table.values, aspect="auto", cmap=USAA_CMAP, interpolation="nearest")
            plt.xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
            plt.yticks(range(len(table.index)), table.index)
            plt.colorbar(label="Number of Articles")
            plt.title("Fraud Trends by Year")
            plt.xlabel("Trend")
            plt.ylabel("Year")
            plt.tight_layout()
            plt.savefig("trend_by_year_heatmap.png", dpi=260)
            print("Saved: trend_by_year_heatmap.png")

if __name__ == "__main__":
    main()
