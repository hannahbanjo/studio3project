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

data_input_path = "all-websites.json"
cache_path = "fraud_results_optimized.json"
data_output_path = "fraud_results_final.json" 

FRAUD_THRESHOLD_HIGH = 0.70 
FRAUD_THRESHOLD_LOW = 0.35

genai_client = genai.Client(api_key=GEMINI_API_KEY)
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load in data
# -----------------------------
print(f"Loading data from {data_input_path}...")
try:
    with open(data_input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except FileNotFoundError:
    raise FileNotFoundError(f"Input file not found at: {data_input_path}")
except json.JSONDecodeError:
    raise ValueError(f"Error decoding JSON from: {data_input_path}. Check file format.")

if "id" not in df.columns:
    raise ValueError("Your dataframe must contain a unique 'id' column.")
print(f"Loaded {len(df)} records.\n")

# -----------------------------
# Load cached results 
# -----------------------------
if os.path.exists(cache_path):
    print("\nLoading cached LLM results…")
    with open(cache_path, "r", encoding="utf-8") as f:
        old_results = json.load(f)

    old_df = pd.DataFrame(old_results)

    df = df.merge(
        old_df[["id", "fraud_related", "fraud_confidence", "fraud_reason", "detection_method"]],
        on="id",
        how="left"
    )

    df["needs_llm"] = df["fraud_related"].isna()

    print(f"Loaded {len(old_df)} cached records.")
    print(f"{df['needs_llm'].sum()} items need Gemini calls.\n")

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
# Clustering
# -----------------------------
k = 3
print(f"\nUsing k={k} clusters")

kmeans = KMeans(n_clusters=k, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(embeddings)

cluster_keywords = {}
for cluster_num in range(k):
    subset = df[df["kmeans_cluster"] == cluster_num]["cleaned_text"].str.lower().str.cat(sep=" ")
    cluster_keywords[cluster_num] = subset[:200]

# -----------------------------
# Embedding fraud prescreening
# -----------------------------
print("\nRunning embedding pre-screening...")

fraud_keywords = [
    "scam alert", "ponzi scheme", "identity theft", "credit card fraud",
    "phishing attack", "cryptocurrency scam", "money laundering", "deceptive practices"
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
Determine if the following article is related to fraud.

Cluster context: "{cluster_label}"

Text:
{text}

Return ONLY JSON exactly in this structure:
{{
  "fraud_related": true/false,
  "fraud_confidence": number between 0 and 1,
  "fraud_reason": "brief explanation",
  "detection_method": "llm_gemini"
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

for col in ["fraud_related", "fraud_confidence", "fraud_reason", "detection_method"]:
    if col not in df.columns:
        df[col] = None

print(f"Starting LLM loop for {df['needs_llm'].sum()} items.")

for i, row in df.iterrows():
    if not row["needs_llm"]:
        continue
    
    cluster_label = cluster_keywords[row["kmeans_cluster"]]
    
    result = classify_with_gemini(row["cleaned_text"], cluster_label)

    df.at[i, "fraud_related"] = result.get("fraud_related")
    df.at[i, "fraud_confidence"] = result.get("fraud_confidence", 0.95)
    df.at[i, "fraud_reason"] = result.get("fraud_reason")
    df.at[i, "detection_method"] = result.get("detection_method")

# -----------------------------
# Save updated JSON cache file
# -----------------------------
results = df.to_dict(orient="records")

with open(cache_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nSaved updated results to {cache_path}")

# -----------------------------
# Save updated final JSON file
# -----------------------------
with open(data_output_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Also saved final results to {data_output_path}")

# -----------------------------
# Add to supabase
# -----------------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

table_name = "scrapeddata"
response = supabase.table(table_name).upsert(results).execute()