# %%
# Imports

from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

# %%
# Get data

filename = "../../word_meanings/00_word_meanings_2.csv"
df = pl.read_csv(filename)
sentences = [row[1] for row in df.iter_rows()]

filename = "../../wordlists/00_all_wordlists.csv"
df = pl.read_csv(filename)
words = [row[1] for row in df.iter_rows()]

# %%
# Setup model

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embedding = model.encode(sentences, convert_to_tensor=False)

# %%
# Compress data

embedded_mds_data = MDS(n_components=2).fit_transform(embedding)
embedded_tsne_data = TSNE(n_components=2, perplexity=5).fit_transform(embedding)


# %%
# Plot the results in regular scatter plot

plt.scatter(embedded_tsne_data[:, 0], embedded_tsne_data[:, 1])
plt.title("Relation between global words of the year")

for i, val in enumerate(embedded_tsne_data):
    plt.annotate(words[i], (val[0], val[1]))

plt.show()

# %%
# Calculate cosine scores
cosine_scores = util.cos_sim(embedding, embedding)

d = {}
for i, v1 in enumerate(words):
    for j, v2 in enumerate(words):
        if i >= j:
            continue
        d[v1 + " vs. " + v2] = cosine_scores[i][j].item()

d_sorted = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
print(d_sorted)

# %%
# Plot the cosine scores in heatmap

fig, ax = plt.subplots()
im = ax.imshow(cosine_scores)
ax.set_xticks(np.arange(len(words)), labels=words)
ax.set_yticks(np.arange(len(words)), labels=words)

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(words)):
    for j in range(len(words)):
        text = ax.text(
            j,
            i,
            (cosine_scores[i, j].item() * 100) // 1,
            ha="center",
            va="center",
            color="w",
        )

ax.set_title("Similarity between the words of the year")
plt.show()

# %%
