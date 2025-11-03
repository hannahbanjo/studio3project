import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import json 
import matplotlib.pyplot as plt
import seaborn as sns

json_data = "/Users/hannahw/studio3project/websites.json"

with open(json_data, 'r') as f:
    data = json.load(f)

cleaned_texts = {item["id"]: item["cleaned_text"] for item in data if "cleaned_text" in item}

# creating embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

texts = list(cleaned_texts.values())
ids = list(cleaned_texts.keys())

embeddings = model.encode(texts)

print(f"Number of sentences: {len(texts)}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Each sentence is represented by a {embeddings.shape[1]}-dimensional vector")
print(f"\nFirst embedding (first 10 dimensions):\n{embeddings[0][:10]}")

# semantic similarity
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

# print("\nSentence Pairs and Similarities")
# print("-" * 70)
# for i in range(len(texts)):
#     for j in range(i+1, len(texts)):
#         sim = similarity_matrix[i][j]
#         print(f"'{texts[i]}' \n vs \n'{texts[j]}'")
#         print(f" Similarity: {sim:.3f}\n")

# visualizing embeddings with PCA
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

print("Legend:")
for i, sentence in enumerate(texts):
    print(f"S{i+1}: {sentence}")
