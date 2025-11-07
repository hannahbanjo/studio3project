from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import json 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from google import genai
import time
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# -------------------
# load in data
# -------------------
json_data = os.environ["JSON_DATA"]

with open(json_data, 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

ids = []
texts = []

for item in data:
    if "cleaned_text" in item and item["cleaned_text"] not in ("", None):
        ids.append(item["id"])
        texts.append(item["cleaned_text"])
    
# -------------------
# creating embeddings
# -------------------
model = SentenceTransformer('All-mpnet-base-v2')

embeddings = model.encode(texts)

# -------------------
# semantic similarity
# -------------------
similarity_matrix = cosine_similarity(embeddings)

plt.figure(figsize=(10, 8))
sns.heatmap(similarity_matrix, 
            annot=True, 
            fmt='.3f', 
            cmap='YlOrRd',
            xticklabels=[f"S{i+1}" for i in range(len(texts))],
            yticklabels=[f"S{i+1}" for i in range(len(texts))],
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('Semantic Similarity Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nSentence Pairs and Similarities")
print("-" * 70)
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        sim = similarity_matrix[i][j]
        print(f"'{ids[i]}' \n vs \n'{ids[j]}'")
        print(f" Similarity: {sim:.3f}\n")

# --------------------------------
# visualizing embeddings with PCA
# --------------------------------
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     s=200, c=range(len(texts)), cmap='viridis', 
                     alpha=0.6, edgecolors='black', linewidth=2)

for i, sentence in enumerate(texts):
    plt.annotate(f"S{i+1}", 
                xy=(embeddings_2d[i, 0], embeddings_2d[i, 1]),
                xytext=(5, 5), textcoords='offset points',
                fontsize=12, fontweight='bold')

plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('2D Visualization of Sentence Embeddings', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ------------------
# kmeans clustering
# ------------------
kmeans = KMeans(n_clusters=3, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(embeddings)
# df = df.merge(df[df["id", "kmeans_cluster"]], on="id", how="left")
k = 3
plt.figure(figsize=(10, 8))
palette = sns.color_palette("husl", k)

sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=df["kmeans_cluster"],
    palette=palette,
    s=150,
    alpha=0.7,
    edgecolor='black'
)

centers_2d = pca.transform(kmeans.cluster_centers_)
plt.scatter(
    centers_2d[:, 0], centers_2d[:, 1],
    c='black', s=300, marker='X', label='Cluster Centers'
)

plt.title(f'KMeans Clusters (k={k}) in PCA Space', fontsize=14, fontweight='bold')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

def generate_word_cloud(cluster_num):
    cluster_docs = df[df["kmeans_cluster"] == cluster_num]["cleaned_text"]
    text = ' '.join(cluster_docs)

    stopwords = set(STOPWORDS)

    wordcloud = WordCloud(width=800, height=400, background_color='white', stopwords=stopwords).generate(text)

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for Cluster {cluster_num}")
    plt.show()

for i in range(k):
    generate_word_cloud(i)

# ------------------------
# Initialize Gemini Client
# ------------------------
client = genai.Client(api_key=GOOGLE_API_KEY)

def detect_fraud(article_text):
    """Classify if text is related to fraud/scams using Gemini."""
    if not isinstance(article_text, str) or len(article_text.strip()) == 0:
        return {"fraud_related": False, "reason": "Empty or invalid text"}

    prompt = f"""
    You are an AI analyst. Determine if the following text relates to Fraud, Scams, or any type of Finacial Misconduct.
    Respond ONLY in JSON with:
    {{
      "fraud_related": true/false,
      "reason": "short explanation"
    }}

    Text:
    {article_text[:15000]}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text_response = response.text.strip()

        json_start = text_response.find("{")
        json_end = text_response.rfind("}") + 1

        return json.loads(text_response[json_start:json_end])

    except Exception as e:
        print("Gemini error:", e)
        return {"fraud_related": None, "reason": str(e)}

# ---------------------------
# Run Gemini for each article
# ---------------------------
df["fraud_related"] = False
df["fraud_reason"] = ""

for i, row in df.iterrows():
    text = str(row.get("cleaned_text", "")).strip()
    cluster = row.get("kmeans_cluster", None)

    if cluster is None:
        continue

    if cluster == 0:
        df.loc[i, "fraud_related"] = True
        df.loc[i, "fraud_reason"] = "Cluster 0 indicates article is fraud-related."
        continue

    if cluster == 1:
        df.loc[i, "fraud_related"] = False
        df.loc[i, "fraud_reason"] = "Cluster 1 indicates article is not fraud-related."
        continue
    
    if cluster == 2:

        print(f"\n Cluster 2 (ambiguous). Sending to LLM --> {row['link'][:60]} ...")
        result = detect_fraud(text)

        df.loc[i, "fraud_related"] = bool(result.get("fraud_related", False))
        df.loc[i, "fraud_reason"] = result.get("reason", "")

        print(" Waiting 7 seconds to avoid rate limits...")
        time.sleep(7)

# Save JSON output
output_data = df.to_dict(orient="records")

with open("fraud_results.json", "w", encoding="utf-8") as f:
    json.dump(df.to_dict("records"), f, indent=2)

# Print total fraud related articles.
fraud_count = df["fraud_related"].sum()
print(f"\n Total articles that are FRAUD-RELATED: {fraud_count}")

print("\n Gemini Analysis complete. Saved results to fraud_results.json")

# -------------------
# Add to Supabase
# -------------------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

records = df.to_dict(orient="records")
table_name = "scrapeddata"
response = supabase.table(table_name).upsert(records).execute()