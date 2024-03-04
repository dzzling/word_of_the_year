# %%
# Imports

from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE, MDS
import matplotlib.pyplot as plt
import numpy as np

# %%
# Get data

sentences = [
    """The term refers to the strong determination and slogan adopted by the Hanshin Tigers baseball team, symbolizing their commitment to achieving victory, as highlighted by their coach's words and the team's performance during the season, ultimately leading to their long-awaited championship win after 38 years.""",
    """The term refers to an advanced artificial intelligence language model developed by OpenAI, which gained significant attention and recognition in Denmark during the year 2023 for its innovative capabilities in natural language processing and conversation generation.""",
    """The term means "AI-generated" in Norwegian, signifying the increasing use and impact of artificial intelligence-generated content and technology in various aspects of society.""",
    """Throughout the year 2023, schoolteachers organised several strikes and demonstrations demanding solutions to problems related to their career progression, working conditions and salaries.""",
    """The term "Krisenmodus" refers to crisis mode and signifies the state of heightened alertness and response mechanisms activated in Germany in the face of various challenges or crises, such as the conflict between ukraine and russia or israel and gaza, the COVID-19 pandemic or economic downturns""",
    """The term means "artificial intelligence" in Russian and its significance reflects the growing interest and development in AI technologies within Russia, impacting various sectors from technology to healthcare.""",
    """The term means "mobilization", reflecting the country's focus on mobilizing resources and efforts, particularly in response to the conflict between Russia and Ukraine""",
    """The word "polarization" has been chosen due to its widespread presence in the media and its evolving significance, referring to situations with two distinctly defined and distant opinions or activities, often implying tension and confrontation.""",
    """The term refers to professional baseball player Seiya Murakami, particularly highlighting his exceptional performance during the season and his status as a baseball deity or hero among fans due to his record-breaking achievements and impactful playing style.""",
    """The term refers to the capital city of Ukraine and gained prominence in Denmark during 2022 due to its association with geopolitical events, particularly the conflict between Ukraine and Russia.""",
    """The term refers to the phenomenon of shrinkflation, where the size or quantity of a product decreases while its price remains the same or increases, reflecting consumer concerns about inflation and product value.""",
    """The Russian invasion of Ukraine in February 2022 gave start to the largest military conflict in Europe since the end of the Second World War; the resulting humanitarian and economic crises, with the compression of the energy and raw materials markets, extended its impact to the whole world.""",
    """The term signifies a pivotal shift or turning point, particularly emphasized by Chancellor Scholz in relation to the Ukrainian conflict, reflecting a significant change or transformation""",
    """The word means "war," and its significance in 2022 reflects the heightened tensions and conflicts due to the war with Ukraine, both domestically and internationally, that shaped Russia's political discourse and actions throughout the year.""",
    """The phrase translates to "Russian military ship, go f*** yourself" and gained significance in 2022 as an expression of defiance and resistance against Russian aggression, particularly during the ongoing conflict between Ukraine and Russia.""",
    """The term refers to artificial intelligence, representing the increasing prominence of AI technologies and their impact on various aspects of society, including innovation, automation, and ethical considerations.""",
]

words = [
    """アレ" (ARE)""",
    "ChatGPT",
    "KI-generert",
    "professor",
    "Krisenmodus",
    "искусственный интеллект",
    "мобілізація",
    "polarization",
    """村神様""",
    "Kyiv",
    "krympflasjon",
    "guerra",
    "Zeitenwende",
    "война",
    "русский военный корабль, иди на хуй",
    "inteligencia artificial",
]

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
