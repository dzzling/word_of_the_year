# %%
# Imports
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
import polars as pl
import numpy as np
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


# %%
# get the word embedding from BERT


def get_word_embedding(word: str):
    input_ids = torch.tensor(tokenizer.encode(word)).unsqueeze(0)
    outputs = model(input_ids)
    last_hidden_states = outputs[
        0
    ]  # The last hidden-state is the first element of the output tuple
    # output[0] is token vector
    # output[1] is the mean pooling of all hidden states
    return last_hidden_states[0][1]


# %%
# Get data

filename = "../../wordlists/00_all_wordlists.csv"
df = pl.read_csv(filename)
words = [row[1] for row in df.iter_rows()]
words_dataframe = pl.DataFrame({"words": words})

filename = "../../wordlists/00_translations.csv"
df = pl.read_csv(filename)
word_translations = [row[1] for row in df.iter_rows()]
translations_dataframe = pl.DataFrame({"Translations": word_translations})

# %%
# Compress dimensions

embedding = []
for word in words:
    embedding.append(get_word_embedding(word).tolist())

embedding_array = np.array(embedding)

embedded_tsne_data_polars = (
    TSNE(n_components=2, perplexity=5)
    .set_output(transform="polars")
    .fit_transform(embedding_array)
)

# %%
# Plot the results with altair

complete_df = pl.concat(
    [embedded_tsne_data_polars, words_dataframe, translations_dataframe],
    how="horizontal",
)

title = alt.TitleParams("Relation between global words of the year", anchor="middle")
plot = (
    alt.Chart(complete_df, title=title)
    .mark_point()
    .encode(
        x=alt.X("tsne0", axis=alt.Axis(labels=False), title=None).scale(
            domain=(-200, 250)
        ),
        y=alt.Y("tsne1", axis=alt.Axis(labels=False), title=None).scale(
            domain=(-100, 80)
        ),
        color=alt.Color("Translations").scale(scheme="tableau20"),
    )
)
text = plot.mark_text(dy=15).encode(text="words")
plot + text

# %%
# Calculate cosine scores
d = {}
for i, v1 in enumerate(words):
    for j, v2 in enumerate(words):
        if i >= j:
            continue
        d[v1 + " vs. " + v2] = cosine_similarity([embedding[i]], [embedding[j]])[0][0]

d_sorted = dict(sorted(d.items(), key=lambda x: x[1], reverse=True))
print(d_sorted)
# %%
