
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
import random

random.seed(42)
np.random.seed(42)

stop_words = get_stop_words('catalan')
# %%
full_train = pd.read_csv("data/train.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
full_test = pd.read_csv("data/test.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
categories = pd.read_csv("data/categories.csv").drop(columns="Unnamed: 0").rename(columns={"Codi QdC": "code", "Títol de entrada del catàleg": "target_title"})

# %%
# IDEAS:
# MAJUSCULES
# STEMMING
# SANT JORDI
# SINGULARS
# TRAIN FULL

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
full_train["title"] = preprocess_str(full_train["title"], stop_words, misspellings)
full_test["title"] = preprocess_str(full_test["title"], stop_words, misspellings)

# %%
def n_match_target(x, set_target):
    set_x = set(x.split())
    return len(set_x & set_target) / len(set_target)

# %%
full_train_cats = full_train.merge(categories, on='code').target_title.unique()
# %%
for i, target in tqdm(enumerate(full_train_cats)):
    set_target = set(target.split())
    # n_match_target("lolo", set_target)
    full_train[f"title_{i}_match"] = full_train.title.apply(lambda x: n_match_target(x, set_target))
    full_test[f"title_{i}_match"] = full_test.title.apply(lambda x: n_match_target(x, set_target))
# %%
full_train.describe()
# %%

train, val = train_test_split(full_train) 
test = full_test.copy()
val["ID"] = val.index
# %%
train_simple = train.loc[:, ["title", "code"]]
full_train_simple = full_train.loc[:, ["title", "code"]]
val_modes = (val.drop(columns="code").merge(train_simple, on='title').groupby("ID").agg({"code": lambda x: x.mode().values[0]}))
test_modes = (test.merge(full_train_simple, on='title').groupby("ID").agg({"code": lambda x: x.mode().values[0]}))

# %%
val = val.merge(val_modes, how='left', on='ID', suffixes=('', '_pred'))

# %%
test = test.merge(test_modes, how='left', on='ID')
# %%
train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)
# %%
val.shape
# %%
train_y = train.code.astype('str')
train_X = train.drop(columns=["code"]).reset_index(drop=True)
test_X = test.drop(columns=["ID", "code"]).reset_index(drop=True)
val_y = val.code.astype('str')
val_X = val.drop(columns=["code", "code_pred", "ID"]).reset_index(drop=True)


# %%
train_X.shape
# %%
transformer_title = Pipeline([
    ("mod1", text.CountVectorizer(max_features=1000)),
    ("dense", MatrixDenser())
])


text_train = transformer_title.fit_transform(train_X.loc[:, "title"])
text_val = transformer_title.transform(val_X.loc[:, "title"])
text_test = transformer_title.transform(test_X.loc[:, "title"])
# %%
train_X_feats = pd.concat([train_X.drop(columns=["title"]), pd.DataFrame(text_train)], axis=1)
val_X_feats = pd.concat([val_X.drop(columns=["title"]), pd.DataFrame(text_val)], axis=1)
test_X_feats = pd.concat([test_X.drop(columns=["title"]), pd.DataFrame(text_test)], axis=1)

# %%
lr = LogisticRegression()
lr.fit(train_X_feats, train_y)
# %%
val_preds_raw = lr.predict(val_X_feats)
val_preds = lr.predict(val_X_feats)
val_preds[val.code_pred.notnull()] = val.code_pred[val.code_pred.notnull()]
val_preds = val_preds.astype("int").astype("str")
print(f1_score(val_preds, val_y, average='micro'))
print(f1_score(val_preds_raw, val_y, average='micro'))
print(f1_score(lr.predict(train_X_feats), train_y, average='micro'))

# %%
# lgb = LGBMClassifier()
# lgb.fit(train_X_feats, train_y)
# print(f1_score(lgb.predict(val_X_feats), val_y, average='micro'))
# print(f1_score(lgb.predict(train_X_feats), train_y, average='micro'))

# %%
test_predictions = lr.predict(test_X_feats)
test_predictions[test.code.notnull()] = test.code[test.code.notnull()]
# %%
test_predictions
# %%
test["ID"]
# %%
submission = test.copy().loc[:, ["ID"]]
# %%
submission["Codi QdC"] = test_predictions.astype("int")
# %%
# submission
submission.to_csv("submissions/05_matching.csv", index=False)

# %%
