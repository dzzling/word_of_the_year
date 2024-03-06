# %%
# Imports

from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import altair as alt

# %%
# Get data

filename = "../../word_meanings/00_word_meanings_2.csv"
df = pl.read_csv(filename)
sentences = [row[1] for row in df.iter_rows()]

filename = "../../wordlists/00_all_wordlists.csv"
df = pl.read_csv(filename)
words = [row[1] for row in df.iter_rows()]

words_dataframe = pl.DataFrame(words)
print(words_dataframe)

# %%
# Setup model

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

embedding = model.encode(sentences, convert_to_tensor=False)

# %%
# Compress data

embedded_mds_data = MDS(n_components=2).fit_transform(embedding)
embedded_tsne_data = TSNE(n_components=2, perplexity=5).fit_transform(embedding)
embedded_tsne_data_polars = (
    TSNE(n_components=2, perplexity=5)
    .set_output(transform="polars")
    .fit_transform(embedding)
)
print(embedded_tsne_data_polars)


# %%
# Plot the results with altair

complete_df = pl.concat([embedded_tsne_data_polars, words_dataframe], how="horizontal")
print(complete_df)

# TODO: Change color (with legend) to translation
title = alt.TitleParams("Relation between global words of the year", anchor="middle")
plot = (
    alt.Chart(complete_df, title=title)
    .mark_point()
    .encode(
        x=alt.X("tsne0", axis=alt.Axis(labels=False), title=None),
        y=alt.Y("tsne1", axis=alt.Axis(labels=False), title=None),
        color=alt.Color("column_0", legend=None),
    )
)
text = plot.mark_text(dy=15).encode(text="column_0")
plot + text

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
# Plot the cosine scores in heatmap with altair
