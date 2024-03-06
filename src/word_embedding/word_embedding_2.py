# %%
# Imports
import gensim.downloader as api
import polars as pl

# %%
# Process data

model = api.load("glove-twitter-200")

filename = "../../wordlists/00_all_wordlists.csv"
df = pl.read_csv(filename)
words = [row[1] for row in df.iter_rows()]
words_dataframe = pl.DataFrame({"words": words})

# %%
# Get similar words
# Error - cannot find in vocabulary
print(model.most_similar("ChatGPT"))

# %%
