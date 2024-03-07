# %%
# Imports

from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE, MDS
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

filename = "../../wordlists/00_translations.csv"
df = pl.read_csv(filename)
word_translations = [row[1] for row in df.iter_rows()]

words_dataframe = pl.DataFrame({"Words": words})
translations_dataframe = pl.DataFrame({"Translations": word_translations})

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


# %%
# Plot the results with altair

complete_df = pl.concat(
    [embedded_tsne_data_polars, words_dataframe, translations_dataframe],
    how="horizontal",
)

# TODO: Change color (with legend) to translation
title = alt.TitleParams("Relation between global words of the year", anchor="middle")
plot = (
    alt.Chart(complete_df, title=title)
    .mark_point()
    .encode(
        x=alt.X("tsne0", axis=alt.Axis(labels=False), title=None),
        y=alt.Y("tsne1", axis=alt.Axis(labels=False), title=None),
        color=alt.Color("Translations"),
    )
)
text = plot.mark_text(dy=15).encode(text="Words")
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

# %%
# Plot the cosine scores in heatmap with altair

x, y = np.meshgrid(words, words)
z = np.ravel(cosine_scores.tolist())

source = pl.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": z})

title = alt.TitleParams("Similarity between the words of the year", anchor="middle")
alt.Chart(source, title=title).mark_rect().encode(
    x=alt.X("x:O", title=None),
    y=alt.Y("y:O", title=None),
    color="z:Q",
)


# %%
