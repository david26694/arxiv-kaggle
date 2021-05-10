
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sktools.matrix_denser import MatrixDenser
import random

random.seed(42)
np.random.seed(42)

pd.set_option('display.max_rows', 500)
stop_words = get_stop_words('catalan')
submit = False
# %%
full_train = pd.read_csv("data/train.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
full_test = pd.read_csv("data/test.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
categories = pd.read_csv("data/categories.csv").drop(columns="Unnamed: 0").rename(columns={"Codi QdC": "code", "Títol de entrada del catàleg": "target_title"})

full_train.code = full_train.code.astype('str')
categories.code = categories.code.astype('str')

full_train.shape

full_title = full_train.title

# %%
spacy_train = pd.read_csv("features/spacy_train.csv")
spacy_test = pd.read_csv("features/spacy_test.csv")
# %%
spacy_train.columns = [f"spacy_{col}" for col in spacy_train.columns]
spacy_test.columns = [f"spacy_{col}" for col in spacy_test.columns]
# %%

full_train = pd.concat([full_train, spacy_train], axis=1)
full_test = pd.concat([full_test, spacy_test], axis=1)
# %%
# IDEAS:
# MAJUSCULES
# STEMMING
# SANT JORDI
# SINGULARS
# ERROR ANALYSIS
# LESS TRAINING

stop_words = stop_words + ['d', 'l', 'al', 'del', 'a', 'dels', 'als', 'deun', 'deuna']

misspellings = [
    ("anul la", "anulla"),
    ("col loca", "colloca"),
    ("tal laci", "tallaci"),
    ("al legac", "allegac"),
    ("parcel lac", "parcellac"),
    ("col labora", "collabora"),
    ("sol lici", "sollici"),
    (" dea", " de a"),
    ("sant jor ", "sant jordi "),
    ("devuit", "18"),
    ("denou", "19"),
    ("resi dencia", "residencia"),
    ("deta ", "data "),
    ("expe deent", "expedient"),
    ("disenv", "desenv"),
    ("disest", "desest"),
    ("mo defic", "modific"),
    ("deputa", "diputa"),
    ("displega", "desplega"),
    ("dispes", "despes"),
    ("discob", "descob"),
    ("deobligacions", "de obligacions"),
    ("disocup", "desocup"),
    ("dehabitatges", "de habitatges"),
    ("regula deres", "reguladores"),
    ("proce dement", "procediment"),
    ("decumenta", "documenta")

]
# %%
stop_words
# %%



def preprocess_str(x, remove_words, misspellings):
    regex_remove = r'\b(?:{})\b'.format('|'.join(remove_words))

    x = x.copy()

    x = x.str.lower()

    x = x.str.replace(r"( S A | SL | sl | SA | s a | sa | SA, | SL,)", " SIGLAS_COMPANYIA ")
    x = x.str.replace(r" 19\d{2}", " YEAR_19")
    x = x.str.replace(r" 20\d{2}", " YEAR_20")
    x = x.str.replace(r"\d{2}/\d{2}/20\d{2}", "DATE")
    x = x.str.replace(r"\d{1,2}:\d{2}", "HOUR")
    x = x.str.replace(r"núm\s+\d+-\d+", " ADDRESS_NUMBER")
    x = x.str.replace(r" c/", " carrer")

    x = x.str.replace(r"[\']", " ' ")
    x = x.str.replace(r"[^\w\s]", ' ')
    for wrong, right in misspellings:
        x = x.str.replace(wrong, right)
    x = x.str.replace(regex_remove, '')
    x = x.str.replace(r"\s+", " ")
    # x = x.str.replace(r"s ", " ")
    # x = x.str.replace(r"s$", "")

    x = x.str.replace(r"(gener |febrer|març|abril|maig|juny|juliol|agost|setembre|octubre|novembre|desembre|deoctubre)", "MONTH ")
    x = x.str.replace(r"( psc | ciu | euia | pp | erc | pp | icv | ciuta dens | deerc | deicv | p decat | comú po dem | ciu denos )", " POLITICAL_PARTY ")
    x = x.str.replace(r"\d{8,}", "LONG_NUMBER")
    x = x.str.replace(r"\d+", "SHORT_NUMBER")

    return x

full_train["title_raw"] = full_train["title"]
full_test["title_raw"] = full_test["title"]

categories["target_title"] = preprocess_str(categories["target_title"], stop_words, misspellings)
full_train["title"] = preprocess_str(full_train["title"], stop_words, misspellings)
full_test["title"] = preprocess_str(full_test["title"], stop_words, misspellings)


full_train.loc[:, ["title", "code"]].merge(categories, how='left', on='code').sort_values(['code', "title"]).drop_duplicates().to_csv("data/train_cleanead.csv", index=False)
# %%
def n_match_target(x, set_target):
    set_x = set(x.split())
    return len(set_x & set_target) / len(set_target)

# %%
full_train_cats = full_train.merge(categories, on='code').target_title.unique()
# %%
full_train.title
# %%
for i, target in tqdm(enumerate(list(full_train_cats))):
    set_target = set(target.split())
    full_train[f"title_{i}_match"] = full_train.title.apply(lambda x: n_match_target(x, set_target))
    full_test[f"title_{i}_match"] = full_test.title.apply(lambda x: n_match_target(x, set_target))
# %%
full_train.loc[:, ["title", "code"]].merge(categories, how='left', on='code').sort_values(['code', "title"]).drop_duplicates().to_csv("data/train_super_cleanead.csv", index=False)
# %%

train, val = train_test_split(full_train, train_size=0.2, stratify=full_train.code)
test = full_test.copy()
val = val.copy()
val["ID"] = val.index
# %%
train_simple = train.loc[:, ["title", "code"]]
full_train_simple = full_train.loc[:, ["title", "code"]]
val_modes = (val.drop(columns="code").merge(train_simple, on='title').groupby("ID").agg({"code": lambda x: x.mode().values[0]}))
test_modes = (test.merge(full_train_simple, on='title').groupby("ID").agg({"code": lambda x: x.mode().values[0]}))

# %%
val = val.merge(val_modes, how='left', on='ID', suffixes=('', '_pred'))
# %%
full_test.copy()
# %%
test = test.merge(test_modes, how='left', on='ID')
# %%
train = train.reset_index(drop=True)
full_train_clean = full_train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)
# %%
val.shape
# %%
train_y = train.code
train_X = train.drop(columns=["code"]).reset_index(drop=True)
full_y = full_train_clean.code
full_X = full_train_clean.drop(columns=["code"]).reset_index(drop=True)
test_X = test.drop(columns=["ID", "code"]).reset_index(drop=True)
val_y = val.code
val_X = val.drop(columns=["code", "code_pred", "ID"]).reset_index(drop=True)


# %%
test_X.loc[:, "title"]
# %%
transformer_title = Pipeline([
    ("mod1", text.CountVectorizer(max_features=1000)),
    ("dense", MatrixDenser())
])

full_train_test = pd.concat([full_X, test_X], axis=0)
transformer_title.fit(full_train_test.loc[:, "title"])

text_train = transformer_title.transform(train_X.loc[:, "title"])
text_full = transformer_title.transform(full_X.loc[:, "title"])
text_val = transformer_title.transform(val_X.loc[:, "title"])
text_test = transformer_title.transform(test_X.loc[:, "title"])
# %%

transformer_title = Pipeline([
    ("mod1", text.CountVectorizer(max_features=500)),
    ("dense", MatrixDenser())
])

full_train_test = pd.concat([full_X, test_X], axis=0)
transformer_title.fit(full_train_test.loc[:, "title_raw"])

text_train_raw = transformer_title.transform(train_X.loc[:, "title_raw"])
text_full_raw = transformer_title.transform(full_X.loc[:, "title_raw"])
text_val_raw = transformer_title.transform(val_X.loc[:, "title_raw"])
text_test_raw = transformer_title.transform(test_X.loc[:, "title_raw"])
# %%
train_X_feats = pd.concat([train_X, pd.DataFrame(text_train), pd.DataFrame(text_train_raw)], axis=1)
full_X_feats = pd.concat([full_X, pd.DataFrame(text_full), pd.DataFrame(text_full_raw)], axis=1)
val_X_feats = pd.concat([val_X, pd.DataFrame(text_val), pd.DataFrame(text_val_raw)], axis=1)
test_X_feats = pd.concat([test_X, pd.DataFrame(text_test), pd.DataFrame(text_test_raw)], axis=1)

# %%
def single_feats(df):

    df = df.copy()
    df["beques_curs"] = df["title"].str.match(r"(beca|beques).*curs").astype('int64')

    df['n_uppercase_letters'] = df['title_raw'].str.count(r'[A-Z]')
    df['n_lowercase_letters'] = df['title_raw'].str.count(r'[a-z]')
    df['n_words'] = df["title_raw"].str.split().str.len()
    df['ratio_letters'] = df['n_uppercase_letters'] / (1 + df["n_lowercase_letters"] + df['n_uppercase_letters'])
    df['ratio_words'] = df['n_uppercase_letters'] / (1 + df["n_words"])
    df["enderrocat"] = df.title.str.match(r"(enderro|endero)").astype('int64')

    return df

train_X_feats = single_feats(train_X_feats).drop(columns=["title", "title_raw"])
full_X_feats = single_feats(full_X_feats).drop(columns=["title", "title_raw"])
val_X_feats = single_feats(val_X_feats).drop(columns=["title", "title_raw"])
test_X_feats = single_feats(test_X_feats).drop(columns=["title", "title_raw"])
# %%
train_X_feats.describe()

# %%
ss = StandardScaler().fit(full_X_feats)

# %%
train_X_feats = ss.transform(train_X_feats)
full_X_feats = ss.transform(full_X_feats)
val_X_feats = ss.transform(val_X_feats)
test_X_feats = ss.transform(test_X_feats)

# %%
pd.DataFrame(train_X_feats).describe()

train_y.describe()
# %%
lr = LogisticRegression(C=0.0075)
lr.fit(train_X_feats, train_y)
lr_cp = lr

# %%
val_preds_raw = lr_cp.predict(val_X_feats)
val_preds = lr_cp.predict(val_X_feats)
val_preds[val.code_pred.notnull()] = val.code_pred[val.code_pred.notnull()]
val_preds = val_preds.astype("int").astype("str")
print(f1_score(val_preds, val_y, average='micro'))
print(f1_score(val_preds_raw, val_y, average='micro'))
print(f1_score(lr_cp.predict(train_X_feats), train_y, average='micro'))

# %%
if submit:
    lr = LogisticRegression(C=0.0075)
    lr.fit(full_X_feats, full_y)
    lr_cp = lr
# %%
test_predictions = lr_cp.predict(test_X_feats)
test_predictions[test.code.notnull()] = test.code[test.code.notnull()]
# %%
test_predictions
# %%
# test.loc[test.title_raw == 'Arbitri de plusvàlua (II de III)', :]
# submission.iloc[10212,:]
# %%
submission = test.copy().loc[:, ["ID"]]
# %%
submission["Codi QdC"] = test_predictions.astype("int")
# %%
if submit:
    submission.to_csv("submissions/11_better_preprocessing.csv", index=False)

# %%

errors = val.loc[val.code.astype("str") != val_preds, :]
errors = errors.merge(categories, on='code', how='left')
errors["pred"] = val_preds[val.code.astype("str") != val_preds].astype('str')
errors = errors.merge(categories.rename(columns={"code": "pred"}), on='pred', how='left')
errors.code.value_counts().to_frame().reset_index().head(20)

# %%
errors.shape
# %%
errors.pred.value_counts().to_frame().reset_index().head(20)

# %%
errors.groupby(["code", "pred"]).size().to_frame().reset_index().sort_values(0).tail(25)
# %%
# errors = val.loc[val.code.astype("str") != val_preds_raw, :]
# errors = errors.merge(categories, on='code', how='left')
# errors["pred"] = val_preds_raw[val.code.astype("str") != val_preds_raw].astype('int64')
# errors = errors.merge(categories.rename(columns={"code": "pred"}), on='pred', how='left')
# print(errors.code.value_counts().head(20))
# print(errors.groupby(["code", "pred"]).size().sort_values())
# %%
# errors.loc[errors.code == '1289', ["title_raw", "code", "pred", "target_title_x", "target_title_y"]]
# %%

# %%
# errors.loc[errors.code == '2947', ["title", "code", "pred", "target_title_x", "target_title_y"]]
# %%

# errors.loc[errors.pred == '2414', ["title_raw", "code", "pred", "target_title_x", "target_title_y"]].tail(100)
# %%
# errors.loc[errors.pred == '1403', ["title_raw", "code", "pred", "target_title_x", "target_title_y"]]

# %%
