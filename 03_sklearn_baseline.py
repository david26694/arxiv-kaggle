
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from stop_words import get_stop_words
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklego.preprocessing import ColumnSelector, ColumnDropper
from sklearn.metrics import f1_score
from sktools.matrix_denser import MatrixDenser

stop_words = get_stop_words('catalan')
# %%
train = pd.read_csv("data/train.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
test = pd.read_csv("data/test.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
categories = pd.read_csv("data/categories.csv").drop(columns="Unnamed: 0").rename(columns={"Codi QdC": "code", "Títol de entrada del catàleg": "target_title"})

# %%
# IDEAS:
# MAJUSCULES
# STEMMING
# SANT JORDI
# SINGULARS
# 

stop_words = stop_words + ['d', 'l', 'al', 'del', 'a', 'dels', 'als', 'deun', 'deuna']

misspellings = {
    "anul la": "anulla",
    "col loca": "colloca",
    "tal laci": "tallaci",
    "al legac": "allegac",
    "parcel lac": "parcellac",
    "col labora": "collabora",
    "sol lici": "sollici",
    " dea": " de a",
    "sant jor ": "sant jordi ",
    "devuit": "18",
    "denou": "19",
    "resi dencia": "residencia",
    "deta ": "data "
}
# %%
stop_words
# %%



def preprocess_str(x, remove_words, misspellings):
    regex_remove = r'\b(?:{})\b'.format('|'.join(remove_words))

    x = x.copy()
    x = x.str.lower()
    x = x.str.replace(r"[\']", " ' ")
    x = x.str.replace(r"[^\w\s]", '')
    for wrong, right in misspellings.items():
        x = x.str.replace(wrong, right)
    x = x.str.replace(regex_remove, '')
    x = x.str.replace(r"\s+", " ")
    return x

categories["target_title"] = preprocess_str(categories["target_title"], stop_words, misspellings)
train["title"] = preprocess_str(train["title"], stop_words, misspellings)
test["title"] = preprocess_str(test["title"], stop_words, misspellings)
# %%
# train.merge(categories, on="code", how='left').to_csv("data/train_cleanead.csv", index=False)
train.merge(categories, on="code", how='left').sort_values("target_title").to_csv("data/train_cleanead.csv", index=False)

# %%
def n_match_target(x, set_target):
    set_x = set(x.split())
    return len(set_x & set_target) / len(set_target)

# %%
train_cats = train.merge(categories, on='code').target_title.unique()
# %%
for i, target in tqdm(enumerate(train_cats)):
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
train_X = train.drop(columns={"code"}).reset_index(drop=True)
test_X = test.drop(columns={"ID"}).reset_index(drop=True)
val_y = val.code.astype('str')
val_X = val.drop(columns={"code"}).reset_index(drop=True)


# %%
train_X.shape
# %%
# pipeline = Pipeline([
#     ("ml_features", FeatureUnion([
#         ("p1", Pipeline([
#             ("grab1", ColumnSelector(columns="title")),
#             ("mod1", text.CountVectorizer(max_features=1000)),
#             ("dense", MatrixDenser())
#         ])),
#         # ("p2", Pipeline([
#         #     ("grab2", ColumnDropper(columns=["title"])),
#         # ]))
#     ])),
#     # ("lr", LogisticRegression())
# ])

transformer_title = Pipeline([
    ("mod1", text.CountVectorizer(max_features=1000)),
    ("dense", MatrixDenser())
])


text_train = transformer_title.fit_transform(train_X.loc[:, "title"])
text_val = transformer_title.transform(val_X.loc[:, "title"])
text_test = transformer_title.transform(test_X.loc[:, "title"])
# %%
pd.DataFrame(text_train)
train_X
# %%
train_X_feats = pd.concat([train_X.drop(columns=["title"]), pd.DataFrame(text_train)], axis=1)
val_X_feats = pd.concat([val_X.drop(columns=["title"]), pd.DataFrame(text_val)], axis=1)
test_X_feats = pd.concat([test_X.drop(columns=["title"]), pd.DataFrame(text_test)], axis=1)

# %%
lr = LogisticRegression()
lr.fit(train_X_feats, train_y)

# %%
print(f1_score(lr.predict(val_X_feats), val_y, average='micro'))
print(f1_score(lr.predict(train_X_feats), train_y, average='micro'))

# %%
# lgb = LGBMClassifier()
# lgb.fit(train_X_feats, train_y)
# print(f1_score(lgb.predict(val_X_feats), val_y, average='micro'))
# print(f1_score(lgb.predict(train_X_feats), train_y, average='micro'))

# %%
test_predictions = lr.predict(test_X_feats)
# %%
test["ID"]
# %%
submission = test.copy().loc[:, ["ID"]]
# %%
submission["Codi QdC"] = test_predictions
# %%
submission.to_csv("submissions/04_better_misspellings.csv", index=False)

# %%
