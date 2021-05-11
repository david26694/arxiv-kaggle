
# %%
import pandas as pd
import numpy as np
from stop_words import get_stop_words
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import random

## Import keras
from keras import Sequential
from keras.layers import Embedding, SpatialDropout1D, LSTM, Dense, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

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
    ("decumenta", "documenta"),
    # ("deobra", "de obra"),
    # ("deocupa", "de ocupa"),
    ("e defi", "edifi"),
    # ("deinforme", "de informe"),
    # ("deincoa", "de incoa"),
    # ("deinfra", "de infra"),
    (" dein", " de in"),
    (" deo", "de o"),
    (" denari", "dinari"),
    ("menja der", "menjador"),
    (" ders", "dors"),
    (" deres", "dores"),
    ("ce der ", "cedir "),
    ("ci der ", "cidir "),
    ("u der", "udir"),
    ("mone der", "moneder"),
    (" der ", "dor ")

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
    x = x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')


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
class2idx = {cat: i for i, cat in enumerate(full_y.unique())}

# %%
idx2class = {v: k for k, v in class2idx.items()}

# %%
train_y = train_y.replace(class2idx)
val_y = val_y.replace(class2idx)
full_y = full_y.replace(class2idx)

# %%
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(full_X.title)

X_train = tokenizer.texts_to_sequences(train_X.title)
X_val = tokenizer.texts_to_sequences(val_X.title)
X_test = tokenizer.texts_to_sequences(test_X.title)

vocab_size = len(tokenizer.word_index) + 1
# Adding 1 because of reserved 0 index

# Sentences
maxlen = 15

X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# %%
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(full_X.title_raw)

X_train_raw = tokenizer.texts_to_sequences(train_X.title_raw)
X_val_raw = tokenizer.texts_to_sequences(val_X.title_raw)
X_test_raw = tokenizer.texts_to_sequences(test_X.title_raw)

vocab_size_raw = len(tokenizer.word_index) + 1
# Adding 1 because of reserved 0 index


X_train_raw = pad_sequences(X_train_raw, padding='post', maxlen=maxlen)
X_val_raw = pad_sequences(X_val_raw, padding='post', maxlen=maxlen)
X_test_raw = pad_sequences(X_test_raw, padding='post', maxlen=maxlen)


# %%s
train_spacy_feats = train_X.filter(regex=r"^spacy.*", axis=1)
val_spacy_feats = val_X.filter(regex=r"^spacy.*", axis=1)
test_spacy_feats = test_X.filter(regex=r"^spacy.*", axis=1)

# %%
# Neural net
embedding_dim = 200

# %%
title_model = Sequential()
title_model.add(
    Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=maxlen)
)
# model.add(SpatialDropout1D(0.5))
title_model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))

# %%
raw_model = Sequential()
raw_model.add(
    Embedding(
        input_dim=vocab_size_raw, 
        output_dim=embedding_dim, 
        input_length=maxlen)
)
# model.add(SpatialDropout1D(0.5))
raw_model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))


# %%
spacy_model = Sequential()
spacy_model.add(Dense(100))

# %%
model = Sequential()
model.add(
    Embedding(
        input_dim=vocab_size, 
        output_dim=embedding_dim, 
        input_length=maxlen)
)
# model.add(SpatialDropout1D(0.5))
model.add(LSTM(100, dropout=0.5, recurrent_dropout=0.5))
model.add(BatchNormalization())
model.add(Dense(len(class2idx), activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              )
model.summary()

# %%
dummy_train_y = to_categorical(train_y)
dummy_val_y = to_categorical(val_y)

# %%
# Train neural net
history = model.fit(
    X_train,
    dummy_train_y,
    batch_size=64,
    epochs=10,
    validation_data=(X_val, dummy_val_y),
    verbose=2
)
# %%
val_preds_wide = model.predict(X_val)
train_preds_wide = model.predict(X_train)
# %%
val_preds_long = np.argmax(val_preds_wide, axis=1)
train_preds_long = np.argmax(train_preds_wide, axis=1)
val_preds_long
# %%
val_y
# %%
print(f1_score(val_preds_long, val_y, average='micro'))
# print(f1_score(val_preds, val_y, average='micro'))
# print(f1_score(lr_cp.predict(train_X_feats), train_y, average='micro'))
# %%
print(f1_score(train_preds_long, train_y, average='micro'))

# %%
# lgb = LGBMClassifier()
# lgb.fit(train_X_feats, train_y)
# print(f1_score(lgb.predict(val_X_feats), val_y, average='micro'))
# print(f1_score(lgb.predict(train_X_feats), train_y, average='micro'))

if submit:
    lr = LogisticRegression()
    lr.fit(full_X_feats, full_y)
    lr_cp = lr
# %%
test_predictions = lr_cp.predict(test_X_feats)
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
if submit:
    submission.to_csv("submissions/07_small_iterations.csv", index=False)

# %%

errors = val.loc[val.code.astype("str") != val_preds, :]
errors = errors.merge(categories, on='code', how='left')
errors["pred"] = val_preds[val.code.astype("str") != val_preds].astype('str')
errors = errors.merge(categories.rename(columns={"code": "pred"}), on='pred', how='left')
print(errors.code.value_counts().head(20))
print(errors.groupby(["code", "pred"]).size().sort_values().to_frame().tail(100))
# %%
# errors = val.loc[val.code.astype("str") != val_preds_raw, :]
# errors = errors.merge(categories, on='code', how='left')
# errors["pred"] = val_preds_raw[val.code.astype("str") != val_preds_raw].astype('int64')
# errors = errors.merge(categories.rename(columns={"code": "pred"}), on='pred', how='left')
# print(errors.code.value_counts().head(20))
# print(errors.groupby(["code", "pred"]).size().sort_values())
# %%
errors.loc[errors.code == 1289, ["title", "code", "pred", "target_title_x", "target_title_y"]]
# %%

# %%
errors.loc[errors.code == 2413, ["title", "code", "pred", "target_title_x", "target_title_y"]]
# %%
