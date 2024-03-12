# %%
# Imports
import gensim.downloader as api
import polars as pl

# %%
# Get model

model = api.load("glove-twitter-200")

# %%
# Process data

filename = "../../wordlists/00_all_wordlists.csv"
df = pl.read_csv(filename)
words = [row[8] for row in df.iter_rows()]
words_dataframe = pl.DataFrame({"words": words})

# %%
# Get similar words
# Error - cannot find in vocabulary
for word in words:
    print(model.most_similar(word))

# %%
