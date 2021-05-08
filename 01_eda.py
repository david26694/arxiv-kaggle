
# %%
import spacy
import pandas as pd
import numpy as np

# %%
nlp = spacy.load("ca_fasttext_wiki_md")

# %%
train = pd.read_csv("data/train.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
test = pd.read_csv("data/test.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
categories = pd.read_csv("data/categories.csv").drop(columns="Unnamed: 0").rename(columns={"Codi QdC": "code"})
# %%
train
# %%
train.merge(categories)
# %%

print(len(train["code"].unique()))
print(len(categories["code"].unique()))

# %%
print(train)
# %%
doc = nlp(train["title"][1100])
# %%
doc.text
# %%
[
    word 
    for word in doc 
    if not word.is_stop 
    and word.text != "d'" 
    and word.text != ':'
    and word.text != ' '
]
# %%

def get_text_emb(question, nlp):
    """Gets average embedding of a sentence, for the non-stop words
    """
    doc = nlp(question)

    # Apparently, name is an stop word in spacy. We are interested
    # in this word, so we'll make an exception.
    vecs = np.array([
        word.vector
        for word in doc
        if not word.is_stop 
        and word.text != "d'" 
        and word.text != ':'
        and word.text != ' '

    ])
    if len(vecs) > 0:
        mean_vec = vecs.mean(axis=0)
        return mean_vec
    else:
        # If there's only stop words, just look for the entire sentence in the vocabulary
        # (almost surely, this will return a 0 vector)
        return nlp.vocab[question].vector

# %%
train_embeddings = [
    get_text_emb(doc, nlp) for doc in train.title
]
# %%
test_embeddings = [
    get_text_emb(doc, nlp) for doc in test.title
]

# %%
categories_embeddings = [
    get_text_emb(doc, nlp) for doc in categories.["Títol de entrada del catàleg"]
]

# %%
