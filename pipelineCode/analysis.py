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
from dotenv import load_dotenv
import os
from google import genai

load_dotenv()

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
    "bold_usaa", [USAA_NAVY, USAA_GOLD]
)

# ----------
# Setup
# ---------- 
JSON_PATH = os.getenv("JSON_DATA")
TEXT_COL = "cleaned_text"
CLASS_COL = "fraud_related"
REASON_COL = "fraud_reason"
DATE_COL = "date"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
CHART_DIR = Path("charts")
CHART_DIR.mkdir(exist_ok=True)

EXTRA_STOPS = {
    "http","https","www","com","org","gov","edu","php","html","amp"
}

JARGON_STOPS = {
    "comptroller", "fdic", "office", "banks", "bank", "organization", "currency", 
    "management", "custody", "occ", "reputation", "risk", "sales", "enforcement", 
    "supervisory", "sanctions", "corrective", "bsa", "supervised", "products", 
    "services", "perspective", "determination", "fall", "compliance", "examiner", 
    "charge", "misconduct", "assessment", "program", "laws", "regulations", 
    "shall", "federal", "policies", "procedures", "profile", "respondent", 
    "russ", "prior", "written", "consumer", "unsafe", "unsound", "specified", 
    "act", "regulation", "board", "quarter", "health", "action", "consent", 
    "applicable", "cease", "desist", "periods", "pursuant", "plan", "set", 
    "forth", "provided", "internal", "controls", "promptly", "committee", 
    "order", "civil", "money", "interagency", "guidance", "savings", 
    "association", "root", "cause", "search", "capital", "liquidity", "stress", 
    "contingency", "credit", "loss", "dodd", "frank", "truth", "lending", "equal", 
    "opportunity", "real", "estate", "mortgage", "servicing", "home", "ownership", 
    "protection", "bureau", "cfpb", "glba", "data", "privacy", "rule", "financial", 
    "corporate", "governance", "directors", "audit", "accounting", "standards", 
    "reporting", "asset", "quality", "earnings", "deposit", "insurance", "fund", 
    "expedited", "availability", "community", "reinvestment", "cra", "fair", 
    "housing", "hmda", "disclosure", "cybersecurity", "information", "third", 
    "party", "vendor", "business", "continuity", "disaster", "recovery", "reserve",
    "united", "states", "secretary", "treasury", "director", "chairman", "acting",
    "ofac"
}

CUSTOM_STOP_WORDS = ENGLISH_STOP_WORDS.union(EXTRA_STOPS).union(JARGON_STOPS)

# ----------
# LOB Keywords
# ----------
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
# ENHANCED Fraud Trend Mapping (ordered by priority)
# ----------
CATEGORIES = [
    # Payment Methods
    "P2P & Digital Payment Fraud",
    "Wire Transfer Fraud",
    "Card Fraud",
    "Check Fraud",
    "ACH & Direct Deposit Fraud",
    
    # Account & Identity
    "Account Takeover",
    "New Account Fraud",
    "Identity Theft & Synthetic Identity",
    "Credential Compromise",
    
    # Social Engineering
    "Phishing & Email Fraud",
    "Vishing & Phone Scams",
    "Impersonation & Spoofing",
    "Business Email Compromise",
    
    # Investment Schemes
    "Cryptocurrency Fraud",
    "Investment Scams & Ponzi Schemes",
    "Securities Fraud",
    
    # Lending & Real Estate
    "Mortgage & Home Equity Fraud",
    "Loan Application Fraud",
    "Real Estate Title & Deed Fraud",
    
    # Specialized
    "Elder Financial Exploitation",
    "Money Mule & Money Laundering",
    "Merchant & Chargeback Fraud",
]

TREND_MAP = [
    # P2P & Digital Payment Fraud
    (r"\b("
     r"p2p|peer.?to.?peer|"
     r"zelle|venmo|paypal|cash app|cashapp|apple pay|google pay|"
     r"rtp|real.?time payment|instant payment|quick pay|"
     r"unauthorized p2p|fake seller|fake buyer|goods not received"
     r")\b", "P2P & Digital Payment Fraud"),
    
    # Wire Transfer Fraud
    (r"\b("
     r"wire transfer|wire fraud|incoming wire|outgoing wire|"
     r"international wire|domestic wire|swift|"
     r"wire redirect|wire diversion|fraudulent wire|unauthorized wire|"
     r"wire scam|wire request"
     r")\b", "Wire Transfer Fraud"),
    
    # Card Fraud 
    (r"\b("
     r"credit card|debit card|card.?not.?present|cnp|card.?present|"
     r"skimming|card skimmer|shimming|"
     r"counterfeit card|cloned card|card cloning|"
     r"unauthorized charge|fraudulent charge|unauthorized transaction|"
     r"lost card|stolen card|card theft"
     r")\b", "Card Fraud"),
    
    # Check Fraud
    (r"\b("
     r"check(?!list|out)|checks|"
     r"check kiting|kiting scheme|"
     r"forged check|altered check|counterfeit check|fake check|"
     r"check washing|check alteration|"
     r"duplicate deposit|double deposit|rdc fraud|remote deposit capture"
     r")\b", "Check Fraud"),
    
    # ACH & Direct Deposit Fraud 
    (r"\b("
     r"ach|automated clearing house|"
     r"direct deposit|payroll redirect|payroll diversion|"
     r"unauthorized ach|ach return|ach fraud|"
     r"recurring payment fraud|subscription fraud"
     r")\b", "ACH & Direct Deposit Fraud"),
    
    # Account Takeover
    (r"\b("
     r"account takeover|ato|acct takeover|"
     r"compromised account|hacked account|hijacked account|"
     r"unauthorized access|unauthorized login|suspicious login|"
     r"session hijack|session theft|cookie theft|token theft"
     r")\b", "Account Takeover"),
    
    # New Account Fraud 
    (r"\b("
     r"new account fraud|account creation fraud|account opening fraud|"
     r"fraudulent application|fake account|"
     r"account bust.?out|bust.?out fraud|"
     r"first party fraud(?!.*elder)"
     r")\b", "New Account Fraud"),
    
    # Identity Theft & Synthetic Identity
    (r"\b("
     r"identity theft|identity fraud|id theft|id fraud|stolen identity|"
     r"synthetic identity|synthetic id|frankenstein fraud|"
     r"fabricated identity|manipulated identity"
     r")\b", "Identity Theft & Synthetic Identity"),
    
    # Credential Compromise
    (r"\b("
     r"credential theft|credential stuffing|credential compromise|"
     r"password spray|password attack|brute force|"
     r"stolen password|stolen credentials|leaked credentials|"
     r"data breach|breach notification|"
     r"username.?password|login.?credentials"
     r")\b", "Credential Compromise"),
    
    # Phishing & Email Fraud
    (r"\b("
     r"phishing|spear phishing|email phishing|"
     r"fraudulent email|fake email|scam email|"
     r"malicious link|suspicious link|fraudulent link|"
     r"credential harvesting|harvest credentials|"
     r"fake website|spoofed website|look.?alike domain|typosquatting"
     r")\b", "Phishing & Email Fraud"),
    
    # Vishing & Phone Scams
    (r"\b("
     r"vishing|voice phishing|phone scam|phone fraud|"
     r"spoof call|spoofed call|spoofed number|caller id spoof|"
     r"robocall|automated call|fake call|"
     r"social security scam|irs scam|tax scam|"
     r"tech support scam|refund scam|warranty scam"
     r")\b", "Vishing & Phone Scams"),
    
    # Impersonation & Spoofing
    (r"\b("
     r"impersonation(?!.*elder)|impersonate|impersonating|"
     r"spoofing(?!.*email)|spoofed domain|domain spoofing|"
     r"fake identity|false identity|assumed identity|"
     r"pretexting|social engineering attack|"
     r"romance scam|catfish|online dating scam"
     r")\b", "Impersonation & Spoofing"),
    
    # Business Email Compromise
    (r"\b("
     r"business email compromise|bec|email compromise|"
     r"ceo fraud|executive impersonation|"
     r"vendor impersonation|supplier impersonation|"
     r"invoice fraud|fake invoice|fraudulent invoice|payment redirect"
     r")\b", "Business Email Compromise"),
    
    # Cryptocurrency Fraud
    (r"\b("
     r"crypto|cryptocurrency|bitcoin|btc|ethereum|eth|altcoin|"
     r"wallet address|crypto wallet|digital wallet|"
     r"nft scam|token scam|coin scam|"
     r"rug pull|exit scam|fake exchange|fake wallet|"
     r"crypto romance|pig butchering|crypto investment scam"
     r")\b", "Cryptocurrency Fraud"),
    
    # Investment Scams & Ponzi Schemes
    (r"\b("
     r"ponzi|pyramid scheme|multi.?level marketing|mlm|"
     r"investment scam|fake investment|fraudulent investment|"
     r"pump and dump|pump.?and.?dump|stock manipulation|"
     r"high.?yield investment|hyip|guaranteed return|"
     r"affinity fraud|advance fee fraud"
     r")\b", "Investment Scams & Ponzi Schemes"),
    
    # Securities Fraud
    (r"\b("
     r"unregistered securities|unlicensed broker|"
     r"securities fraud|insider trading|market manipulation|"
     r"microcap fraud|penny stock|"
     r"promissory note fraud|private placement fraud"
     r")\b", "Securities Fraud"),
    
    # Mortgage & Home Equity Fraud
    (r"\b("
     r"mortgage|mortgage fraud|mortgage scam|"
     r"heloc|home equity|home equity line|"
     r"refinance scam|refi scam|refinance fraud|"
     r"reverse mortgage scam|"
     r"foreclosure rescue|foreclosure scam|loan modification scam"
     r")\b", "Mortgage & Home Equity Fraud"),
    
    # Loan Application Fraud
    (r"\b("
     r"loan application fraud|loan fraud|loan scam|"
     r"loan stacking|stacked loans|"
     r"income fraud|income misrepresentation|employment fraud|"
     r"document fraud|forged documents|fake paystub|fake w2|"
     r"ppp fraud|paycheck protection|eidl fraud|"
     r"auto loan fraud|personal loan fraud|student loan fraud"
     r")\b", "Loan Application Fraud"),
    
    # Real Estate Title & Deed Fraud
    (r"\b("
     r"title fraud|deed fraud|property fraud|"
     r"forged deed|fake deed|fraudulent deed|"
     r"property theft|real estate theft|home title theft|"
     r"quit.?claim deed fraud|short sale fraud"
     r")\b", "Real Estate Title & Deed Fraud"),
    
    # Elder Financial Exploitation
    (r"\b("
     r"elder abuse|elder scam|elder fraud|elder exploitation|"
     r"senior scam|senior fraud|senior exploitation|senior abuse|"
     r"elder financial exploitation|financial exploitation.*elder|"
     r"exploitation.*elderly|abuse.*older adult|grandparent scam|"
     r"caregiver fraud|power of attorney abuse|poa abuse|"
     r"nursing home fraud|assisted living fraud"
     r")\b", "Elder Financial Exploitation"),
    
    # Money Mule & Money Laundering
    (r"\b("
     r"money mule|mule account|witting mule|unwitting mule|"
     r"money laundering|laundering|layering|structuring|smurfing|"
     r"funnel account|drop account|shell account|"
     r"rapid movement|pass.?through account|"
     r"suspicious activity|sar|suspicious transaction"
     r")\b", "Money Mule & Money Laundering"),
    
    # Merchant & Chargeback Fraud
    (r"\b("
     r"chargeback|charge.?back|dispute|merchant dispute|"
     r"friendly fraud|first.?party fraud.*chargeback|"
     r"refund fraud|return fraud|"
     r"merchant fraud|payment processor fraud|"
     r"triangulation fraud|reshipping"
     r")\b", "Merchant & Chargeback Fraud"),
]

def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def clean_text_strong(text):
    if not isinstance(text, str): 
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) 
    text = re.sub(r"<[^>]+>", " ", text) 
    
    junk_phrases = [
        "skip to main content", "main content", "skip navigation", "privacy policy",
        "terms of service", "terms of use", "cookie policy", "footer", "header",
        "home menu", "navigation menu", "menu", "sidebar", "log in", "login",
        "sign in", "read more", "click here", "copyright", "all rights reserved",
        "proposed rule", "financial institution", "united states", "federal reserve", 
        "bank secrecy act", "bank secrecy", "helpwithmybank", "office comptroller currency", 
        "risk management", "sales practices", "enforcement actions",
        "supervisory objection", "corrective actions", "bsa compliance", 
        "supervised institutions", "products services", "semiannual risk", 
        "written determination", "fall 2024", "perspective fall", "compliance committee",
        "examiner-in charge", "misconduct problem", "risk assessment",
        "compliance program", "laws regulations", "shall include",
        "federal ing", "policies procedures", "bsa risk", "risk profile",
        "respondent tolstedt", "respondent strother", "respondent julian",
        "respondent russ", "russ anderson", "prior written", "examiner-in",
        "sales goals", "ngi", "trade commission",
        "fraud prevention", "2024", "wells fargo", "incentive compensation",
        "news release", "transaction monitoring", "vital signs", "challenge", 
        "comptroller", "board directors", "civil money penalty", "consent order",
        "cease desist", "written agreement", "money penalty", "soundness",
        "department", "total", "look-back", "bankers", "adhere", 'official', "government",
        "careers", "handbook", "quick", "billion", "website", "application", "contact",
        "requested", "michael"
    ]

    for jp in junk_phrases: 
        text = text.replace(jp, " ")
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text):
    text = clean_text_strong(text)
    words = re.findall(r"\b[\w'-]+\b", text.lower())
    words = [w for w in words if w not in CUSTOM_STOP_WORDS and len(w) > 2]
    return " ".join(words)

def extract_ngrams(text, n=1):
    words = text.split()
    if n == 1:
        return Counter(words)
    return Counter(" ".join(words[i:i+n]) for i in range(len(words) - n + 1))

def gemini_classify(text):
    prompt = f"""
    Classify the following fraud description into exactly one of these categories:
    {CATEGORIES}

    Fraud text: "{text}"

    Return ONLY the category string. No explanation.
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    label = response.text.strip()

    # safety catch: if Gemini returns something weird
    if label not in CATEGORIES:
        label = "Account & Identity Attacks"

    return label

def assign_trend(text):
    if not isinstance(text, str):
        return "Account & Identity Attacks"

    lower = text.lower()

    for pattern, label in TREND_MAP:
        if re.search(pattern, lower):
            return label

    return gemini_classify(text)

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
        df = df[df[CLASS_COL] == True]
    if df.empty:
        print("No fraud-related articles found.")
        return

    # K-Means Cluster Visualization using original embeddings
    if "embedding" in df.columns and "kmeans_cluster" in df.columns:
        print("Plotting K-Means clusters based on original embeddings...")
        
        df_clustered = df[df["embedding"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)].copy()

        if df_clustered.empty:
            print("No valid embeddings found for clustering.")
        else:
            embedding_data = np.stack(df_clustered["embedding"])
            clusters = df_clustered["kmeans_cluster"].values
            
            if len(np.unique(clusters)) > 1:
                pca = PCA(n_components=2)
                components = pca.fit_transform(embedding_data)

                plt.figure(figsize=(10, 8))
                base_colors = [USAA_NAVY, USAA_GOLD, USAA_BABY_BLUE, USAA_SLATE, USAA_LIGHT_GRAY]
                for i, cluster_id in enumerate(np.unique(clusters)):
                    idx = clusters == cluster_id
                    plt.scatter(components[idx, 0], components[idx, 1],
                                c=base_colors[i % len(base_colors)],
                                label=f"Cluster {cluster_id}", alpha=0.7, edgecolors="k", linewidth=0.3)
                plt.legend()
                plt.title("K-Means Clusters (PCA 2D) - Original Embeddings")
                plt.tight_layout()
                plt.savefig(CHART_DIR/"kmeans_clusters_original_embeddings.png", dpi=300)
                print("Saved: charts/kmeans_clusters_original_embeddings.png")
            else:
                print("Skipping K-Means plot: Only one unique cluster found.")
        
    df = df_clustered.copy()
    if df.empty:
        print("No data remaining after embedding check.")
        return
        
    print("\nApplying aggressive cleaning and stopword removal for text analysis...")
    df["fully_cleaned"] = df[TEXT_COL].astype(str).apply(preprocess_text)

    combined_text = " ".join(df["fully_cleaned"])

    # Bigrams
    bigrams = extract_ngrams(combined_text, n=2)
    top_bigrams = bigrams.most_common(10)
    print("\nTop 10 Bigrams:")
    for term, freq in top_bigrams:
        print(f"{term}: {freq}")

    # LOB Detection
    df["LOB"] = df[TEXT_COL].astype(str).apply(detect_lob)
    print("\nLOB Distribution:\n", df["LOB"].value_counts())

    # Trend Detection
    if REASON_COL in df.columns:
        df["trend"] = df[REASON_COL].apply(assign_trend)
        trend_counts = df["trend"].value_counts()
        print("\n=== ENHANCED TREND DISTRIBUTION ===")
        print(trend_counts)
        
        other_reasons = df[df["trend"] == "Other"][REASON_COL].head(10)
        if len(other_reasons) > 0:
            print("\nSample 'Other' fraud reasons:")
            for reason in other_reasons:
                print(f"  - {reason}")

    # Plot top bigrams
    plt.figure(figsize=(12, 7))
    terms, counts = zip(*top_bigrams[:10])
    plt.barh(range(len(terms)-1, -1, -1), counts, color=USAA_NAVY, edgecolor=USAA_SLATE)
    plt.yticks(range(len(terms)), terms)
    plt.xlabel("Frequency")
    plt.title("Top 10 Bigrams")
    plt.tight_layout()
    plt.savefig(CHART_DIR/"top_clean_bigrams.png", dpi=300, bbox_inches="tight")
    print("Saved: charts/top_clean_bigrams.png")

    # Top trends by LOB
    lob_trends = {}
    for lob, rows in df.groupby("LOB"):
        if REASON_COL in rows.columns:
            trends = rows[REASON_COL].apply(assign_trend)
            counts = trends.value_counts().head(7)
            lob_trends[lob] = counts

    lob_colors = {"Banking": USAA_NAVY, "Insurance": USAA_GOLD, "Investing": USAA_BABY_BLUE}
    all_trends, all_counts, all_colors = [], [], []

    for lob, counts in lob_trends.items():
        for trend, count in counts.items():
            all_trends.append(trend)
            all_counts.append(count)
            all_colors.append(lob_colors.get(lob, USAA_SLATE))

    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(all_trends)), all_counts, color=all_colors, edgecolor=USAA_SLATE)
    plt.xticks(range(len(all_trends)), all_trends, rotation=45, ha="right")
    handles = [plt.Rectangle((0,0),1,1, color=c, label=l) for l,c in lob_colors.items()]
    plt.legend(handles=handles, title="Line of Business")
    plt.title("Top Fraud Trends by Line of Business")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(CHART_DIR/"lob_trends.png", dpi=300, bbox_inches="tight")
    print("Saved: charts/lob_trends.png")

    # Articles by Year
    if "year" in df.columns and df["year"].notna().any():
        year_counts = df["year"].value_counts().sort_index()
        year_counts = year_counts[year_counts.index != 2000]
        plt.figure(figsize=(10, 5))
        plt.bar(year_counts.index.astype(str), year_counts.values, color=USAA_NAVY)
        plt.title("Fraud-Related Articles by Year")
        plt.ylabel("Count")
        plt.xlabel("Year")
        plt.tight_layout()
        plt.savefig(CHART_DIR/"articles_by_year.png", dpi=300)
        print("Saved: charts/articles_by_year.png")

    # Word Clouds per Cluster
    if "kmeans_cluster" in df.columns:
        print("\n=== Generating word clouds per cluster ===")
        os.makedirs("wordclouds", exist_ok=True)

        unique_clusters = sorted(df["kmeans_cluster"].unique())
        print(f"Found clusters: {unique_clusters}\n")

        for cluster_num in unique_clusters:
            cluster_df = df[df["kmeans_cluster"] == cluster_num]

            # ---- DEBUG SUMMARY ----
            raw_text = " ".join(cluster_df["fully_cleaned"])
            print(
                f"Cluster {cluster_num}: {len(cluster_df)} rows | "
                f"has_text={bool(raw_text.strip())} | "
                f"has_subclusters={ 'subcluster' in df.columns and cluster_df['subcluster'].notna().any() }"
            )
            if not raw_text.strip():
                print(f"  Skipping cluster {cluster_num} (empty text)")
                continue

            wc = WordCloud(
                width=1200, height=600, background_color="white",
                colormap=BOLD_USAA_CMAP, max_words=120, min_word_length=3,
                contour_width=1, contour_color=USAA_SLATE,
                stopwords=CUSTOM_STOP_WORDS, random_state=42
            ).generate(raw_text)

            plt.figure(figsize=(12, 6))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(
                f"Cluster {cluster_num} – Key Themes",
                fontsize=18, color=USAA_NAVY, pad=30
            )
            plt.tight_layout(pad=0)

            filename = f"charts/wordclouds/cluster_{cluster_num}_wordcloud.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"  → Saved {filename}")


    # Trend by Year Heatmap
    if "year" in df.columns and REASON_COL in df.columns:
        valid = df[df["year"].notna()].copy()
        table = pd.crosstab(valid["year"], valid["trend"])
        if not table.empty:
            plt.figure(figsize=(14, 8))
            plt.imshow(table.values, aspect="auto", cmap=USAA_CMAP)
            plt.colorbar(label="Number of Articles")
            plt.xticks(range(len(table.columns)), table.columns, rotation=45, ha="right")
            plt.yticks(range(len(table.index)), table.index)
            plt.title("Fraud Trend Evolution Over Time")
            plt.xlabel("Trend")
            plt.ylabel("Year")
            plt.tight_layout()
            plt.savefig(CHART_DIR/"trend_by_year_heatmap.png", dpi=300)
            print("Saved: charts/trend_by_year_heatmap.png")
    
if __name__ == "__main__":
    main()