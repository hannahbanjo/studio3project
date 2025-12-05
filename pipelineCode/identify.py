import os
import json
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from google import genai
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import time
from supabase import Client, create_client
# -----------------------------
# Set up
# -----------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

JSON_DATA = os.getenv("JSON_DATA")
CACHE_PATH = os.getenv("CACHE_PATH")

FRAUD_THRESHOLD_HIGH = 0.70 
FRAUD_THRESHOLD_LOW = 0.35

genai_client = genai.Client(api_key=GEMINI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load in data
# -----------------------------
print(f"Loading data from {JSON_DATA}...")
try:
    with open(JSON_DATA, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found at: {JSON_DATA}")
except json.JSONDecodeError:
    raise ValueError(f"Error decoding JSON from: {JSON_DATA}. Check file format.")

if "id" not in df.columns:
    raise ValueError("Your dataframe must contain a unique 'id' column.")
print(f"Loaded {len(df)} records.\n")

# -----------------------------
# Load cached results 
# -----------------------------
if os.path.exists(CACHE_PATH):
    print("\nLoading cached LLM results…")
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        old_results = json.load(f)

    old_df = pd.DataFrame(old_results)
    
    optional_cols = ["fraud_related", "fraud_confidence", "fraud_reason", "detection_method", "summary"]
    
    for col in optional_cols:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)
    
    cache_cols = ["id"]
    for col in optional_cols:
        if col in old_df.columns:
            cache_cols.append(col)
    
    df = df.merge(
        old_df[cache_cols],
        on="id",
        how="left"
    )

    # If fraud_related exists, mark items that need LLM
    if "fraud_related" in df.columns:
        df["needs_llm"] = df["fraud_related"].isna()
        print(f"Loaded {len(old_df)} cached records.")
        print(f"{df['needs_llm'].sum()} items need Gemini calls.\n")
    else:
        print("Cache exists but no fraud_related column found — will process all items\n")
        df["needs_llm"] = True

else:
    print("No cache found — starting fresh\n")
    df["needs_llm"] = True

# -----------------------------
# Create text embeddings
# -----------------------------
print("Generating embeddings…")

embeddings = model.encode(df["cleaned_text"].fillna("").tolist(), show_progress_bar=True)
df["embedding"] = embeddings.tolist()

print(f"Successfully added embeddings for {len(df)} records to the DataFrame.")

# -----------------------------
# Embedding fraud prescreening
# -----------------------------
print("\nRunning embedding pre-screening...")

fraud_keywords = [
    "bank fraud", "loan fraud", "mortgage fraud", "deposit fraud",
    "wire fraud", "check fraud", "identity theft", "account takeover",
    "payment fraud", "elder financial abuse",
    "phishing scam", "vishing scam", "smishing scam",
    "fake bank website", "spoofed bank call", "imposter scam",
    "advance fee scam", "romance scam", "check kiting",
    "unauthorized withdrawals", "fraudulent transfers",
    "money laundering", "structuring transactions", "suspicious activity",
    "suspicious transaction", "shell company", "terrorist financing",
    "synthetic identity fraud", "kyc fraud", "aml violation",
    "predatory lending", "deceptive practices", "UDAAP violation",
    "fair lending fraud", "loan application misrepresentation",
    "cryptocurrency scam", "crypto investment fraud", "online banking scam",
    "ransomware", "data breach", "account compromise",
    "scam alert", "fraudulent activity", "consumer complaint",
    "fraud investigation", "suspicious pattern", "unauthorized account opening",
    "ACH fraud", "unauthorized ACH transfer", "card fraud", "phishing",
    "impersonation", "synthetic", "AI", "voice cloning", "digital wallet",
    "spoof", "crypto", "impersonation", "malware", "SIM", "employment",
    "email", "extortion", "data breach", "deepfakes", "new account fraud",
    "friendly fraud", "chargeback fraud", "embezzlement", "collusive fraud",
    "account bust-out", "money mule activity", "device spoofing", 
    "business email compromise", "session hijacking", "man-in-the-middle attack"
    "account credential compromise", "SIM swap fraud", "social engineering attack",
    "card skimming", "credential stuffing", "brute force login attempt", 
    "unauthorized ACH transfer", "ATM cash-out scheme", "mobile check deposit fraud",
    "remote deposit capture fraud", "account hijacking", "scam"
]

fraud_embeddings = model.encode(fraud_keywords)
fraud_centroid = np.mean(fraud_embeddings, axis=0).reshape(1, -1) 

# Create embeddings
all_embeddings = np.array(df["embedding"].tolist())
similarity_scores = cosine_similarity(all_embeddings, fraud_centroid).flatten()
df["fraud_score"] = similarity_scores

is_ambiguous = df["fraud_score"].apply(
    lambda x: FRAUD_THRESHOLD_LOW <= x <= FRAUD_THRESHOLD_HIGH
)

df.loc[df["fraud_score"] > FRAUD_THRESHOLD_HIGH, ["fraud_related", "fraud_confidence", "fraud_reason", "detection_method"]] = \
    [True, 1.0, "Pre-screen: High embedding similarity to fraud keywords.", "embedding_high"]

df.loc[df["fraud_score"] < FRAUD_THRESHOLD_LOW, ["fraud_related", "fraud_confidence", "fraud_reason", "detection_method"]] = \
    [False, 1.0, "Pre-screen: Low embedding similarity to fraud keywords.", "embedding_low"]

df["needs_llm"] = is_ambiguous & df["fraud_related"].isna() 


print(f"Pre-screened {len(df) - df['needs_llm'].sum()} items.")
print(f"{df['needs_llm'].sum()} items remaining for Gemini LLM analysis.")

# ------------------------------------------
# Gemini classification for ambiguous cases
# ------------------------------------------
def classify_with_gemini(text, cluster_label, max_retries=5, initial_delay=5):
    prompt = f"""
Determine if the following article is related to fraud and summarize the articles.

Cluster context:
{cluster_label}

Text:
{text}

Return ONLY JSON exactly in this structure:
{{
  "fraud_related": true/false,
  "fraud_confidence": number between 0 and 1,
  "fraud_reason": "brief explanation",
  "detection_method": "llm_gemini",
  "summary": "summary of each article
}}
"""
    for attempt in range(max_retries):
        try:
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config={
                    "response_mime_type": "application/json"
                }
            )
            
            result = json.loads(response.text)
            result["detection_method"] = "llm_gemini"
            return result

        except Exception as e:
            error_str = str(e)
            
            if "503 UNAVAILABLE" in error_str and attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)  
                print(f"Gemini ERROR: 503 UNAVAILABLE. Retrying in {delay} seconds (Attempt {attempt + 1}/{max_retries}).")
                time.sleep(delay)
                continue 

            print("Gemini ERROR:", error_str)
            break 

    return {
        "fraud_related": False,
        "fraud_confidence": 0.0,
        "fraud_reason": "error_rate_limited",
        "detection_method": "error"
    }

print("\nRunning Gemini fraud detection (only ambiguous items)…")

for col in ["fraud_related", "fraud_confidence", "fraud_reason", "detection_method", "summary"]:
    if col not in df.columns:
        df[col] = None

print(f"Starting LLM loop for {df['needs_llm'].sum()} items.")

for i, row in df.iterrows():
    if not row["needs_llm"]:
        continue
    
    cluster_label = ""
    
    result = classify_with_gemini(row["cleaned_text"], cluster_label)

    df.at[i, "fraud_related"] = result.get("fraud_related")
    df.at[i, "fraud_confidence"] = result.get("fraud_confidence", 0.95)
    df.at[i, "fraud_reason"] = result.get("fraud_reason")
    df.at[i, "detection_method"] = result.get("detection_method")
    df.at[i, "summary"] = result.get("summary")

# -----------------------------
# Clustering 
# -----------------------------
print("\n" + "="*50)
print("Clustering Fraud-Related Articles…")
print("="*50)

# Filter to fraud-related articles
fraud_df = df[df["fraud_related"] == True].copy()
print(f"\nFound {len(fraud_df)} fraud-related articles for clustering.")

if len(fraud_df) > 0:
    fraud_embeddings = np.array(fraud_df["embedding"].tolist())
    
    k = min(3, len(fraud_df))
    print(f"Using k={k} clusters for fraud articles")
    
    kmeans = KMeans(n_clusters=k, random_state=42)
    fraud_df["kmeans_cluster"] = kmeans.fit_predict(fraud_embeddings)
    
    cluster_keywords = {}
    for cluster_num in range(k):
        subset = fraud_df[fraud_df["kmeans_cluster"] == cluster_num]["cleaned_text"].str.lower().str.cat(sep=" ")
        cluster_keywords[cluster_num] = subset[:200]
        print(f"\nCluster {cluster_num} ({len(fraud_df[fraud_df['kmeans_cluster'] == cluster_num])} articles):")
        print(f"  Keywords: {cluster_keywords[cluster_num][:100]}...")
    
    df = df.merge(
        fraud_df[["id", "kmeans_cluster"]],
        on="id",
        how="left",
        suffixes=("", "_fraud")
    )
    
    if "kmeans_cluster_fraud" in df.columns:
        df["kmeans_cluster"] = df["kmeans_cluster_fraud"]
        df.drop("kmeans_cluster_fraud", axis=1, inplace=True)
else:
    print("No fraud articles found to cluster.")
    df["kmeans_cluster"] = None

# -----------------------------
# Save updated JSON cache file
# -----------------------------
duplicate_suffixes = ['_x', '_y', '_fraud']
cols_to_drop = [col for col in df.columns if any(col.endswith(suffix) for suffix in duplicate_suffixes)]
if cols_to_drop:
    print(f"\nCleaning up duplicate columns: {cols_to_drop}")
    df.drop(cols_to_drop, axis=1, inplace=True)

results = df.to_dict(orient="records")

for record in results:
    for key, value in record.items():
        if isinstance(value, float):
            if np.isnan(value) or np.isinf(value):
                record[key] = None

with open(CACHE_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved updated results to {CACHE_PATH}")

# -----------------------------
# Save updated final JSON file
# -----------------------------
with open(JSON_DATA, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Also saved final results to {JSON_DATA}")

# -----------------------------
# Add to supabase
# -----------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

table_name = "scrapeddata"
response = supabase.table(table_name).upsert(results).execute()