
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from stop_words import get_stop_words
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

stop_words = get_stop_words('catalan')
# %%
train = pd.read_csv("data/train.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
test = pd.read_csv("data/test.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
categories = pd.read_csv("data/categories.csv").drop(columns="Unnamed: 0").rename(columns={"Codi QdC": "code", "Títol de entrada del catàleg": "target_title"})
# %%
print(stopwords.words('spanish'))
categories
# %%
stop_words = stop_words + ['d', 'l', 'al', 'del', 'a', 'dels', 'als']
# %%
stop_words
# %%



def preprocess_str(x, remove_words):
    regex_remove = r'\b(?:{})\b'.format('|'.join(remove_words))

    x = x.copy()
    x = x.str.lower()
    x = x.str.replace(r"[\']", " ' ")
    x = x.str.replace(r"[^\w\s]", '')
    x = x.str.replace(regex_remove, '')
    return x

categories["target_title"] = preprocess_str(categories["target_title"], stop_words)
train["title"] = preprocess_str(train["title"], stop_words)
test["title"] = preprocess_str(test["title"], stop_words)
# %%
categories

# %%
def n_match_target(x, set_target):
    set_x = set(x.split())
    return len(set_x & set_target) / len(set_target)

# %%
for i, target in tqdm(enumerate(categories.target_title)):
    set_target = set(target.split())
    # n_match_target("lolo", set_target)
    train[f"title_{i}_match"] = train.title.apply(lambda x: n_match_target(x, set_target))
    test[f"title_{i}_match"] = test.title.apply(lambda x: n_match_target(x, set_target))
# %%
train
# %%
train.describe()
# %%

train, val = train_test_split(train) 
# %%
train_y = train.code.astype('str')
train_X = train.drop(columns={"code", "title"})
test_X = test.drop(columns={"title", "ID"})
val_y = val.code.astype('str')
val_X = val.drop(columns={"code", "title"})


# %%
lr = LogisticRegression()
lr.fit(train_X, train_y)
# %%

f1_score(lr.predict(val_X), val_y, average='micro')
# %%
test_predictions = lr.predict(test_X)
# %%
test["ID"]
# %%
submission = test.copy().loc[:, ["ID"]]
# %%
submission["Codi QdC"] = test_predictions
# %%
submission.to_csv("submissions/first_model.csv", index=False)
# %%
