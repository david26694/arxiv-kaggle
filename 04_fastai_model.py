
# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction import text
from stop_words import get_stop_words
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sktools.matrix_denser import MatrixDenser
import random
from sklearn.preprocessing import MinMaxScaler
from utils.rank import GaussRankScaler

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn


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
# TRAIN FULL
# TRANSFORM FULL + TEST
# ERROR ANALYSIS
# LESS TRAINING
# FROM 1, 3 to 1 or 1, 2

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

    x = x.str.replace(r" 19\d{2}", " YEAR_19")
    x = x.str.replace(r" 20\d{2}", " YEAR_20")
    x = x.str.replace(r"\d{2}/\d{2}/20\d{2}", "DATE")
    x = x.str.replace(r"\d{1,2}:\d{2}", "HOUR")
    x = x.str.replace(r"núm\s+\d+-\d+", " ADDRESS_NUMBER")
    x = x.str.replace(r" c/", " carrer")

    x = x.str.replace(r"[\']", " ' ")
    x = x.str.replace(r"[^\w\s]", ' ')
    for wrong, right in misspellings.items():
        x = x.str.replace(wrong, right)
    x = x.str.replace(regex_remove, '')
    x = x.str.replace(r"\s+", " ")
    # x = x.str.replace(r"s ", " ")
    # x = x.str.replace(r"s$", "")

    x = x.str.replace(r"(gener |febrer |març |abril |maig |juny |juliol |agost |setembre |octubre |novembre |desembre |deoctubre )", "MONTH ")
    x = x.str.replace(r"\d{8,}", "LONG_NUMBER")
    x = x.str.replace(r"\d+", "SHORT_NUMBER")

    return x

categories["target_title"] = preprocess_str(categories["target_title"], stop_words, misspellings)
full_train["title"] = preprocess_str(full_train["title"], stop_words, misspellings)
full_test["title"] = preprocess_str(full_test["title"], stop_words, misspellings)


full_train.loc[:, ["title", "code"]].merge(categories, how='left', on='code').sort_values(['code', "title"]).to_csv("data/train_cleanead.csv", index=False)
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
full_train.loc[:, ["title", "code"]].merge(categories, how='left', on='code').sort_values(['code', "title"]).to_csv("data/train_super_cleanead.csv", index=False)
# %%

train, val = train_test_split(full_train, train_size=0.8, stratify=full_train.code)
test = full_test.copy()
# %%
val = val.copy()
val["ID"] = val.index
# val.loc[val.index, ("ID")] = val.index.values
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
val["ID"]
# %%
transformer_title = Pipeline([
    ("mod1", text.CountVectorizer(max_features=1500, ngram_range=(1, 3))),
    ("dense", MatrixDenser())
])

full_train_test = pd.concat([full_X, test_X], axis=0)
transformer_title.fit(full_train_test.loc[:, "title"])

text_train = transformer_title.transform(train_X.loc[:, "title"])
text_full = transformer_title.transform(full_X.loc[:, "title"])
text_val = transformer_title.transform(val_X.loc[:, "title"])
text_test = transformer_title.transform(test_X.loc[:, "title"])

# %%
train_X_feats = pd.concat([train_X.drop(columns=["title"]), pd.DataFrame(text_train)], axis=1)
full_X_feats = pd.concat([full_X.drop(columns=["title"]), pd.DataFrame(text_full)], axis=1)
val_X_feats = pd.concat([val_X.drop(columns=["title"]), pd.DataFrame(text_val)], axis=1)
test_X_feats = pd.concat([test_X.drop(columns=["title"]), pd.DataFrame(text_test)], axis=1)

# %%
full_train_test = pd.concat([full_X_feats, test_X_feats], axis=0)
mn = MinMaxScaler().fit(full_train_test)

# %%

train_X_feats = mn.transform(train_X_feats)
full_X_feats = mn.transform(full_X_feats)
val_X_feats = mn.transform(val_X_feats)
test_X_feats = mn.transform(test_X_feats)

# %%
class2idx = {cat: i for i, cat in enumerate(full_y.unique())}

# %%
train_y = train_y.replace(class2idx)
val_y = val_y.replace(class2idx)
full_y = full_y.replace(class2idx)
# %%


class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


full_dataset = ClassifierDataset(torch.from_numpy(full_X_feats).float(), torch.from_numpy(full_y.values).long())
train_dataset = ClassifierDataset(torch.from_numpy(train_X_feats).float(), torch.from_numpy(train_y.values).long())
val_dataset = ClassifierDataset(torch.from_numpy(val_X_feats).float(), torch.from_numpy(val_y.values).long())
test_dataset = ClassifierDataset(torch.from_numpy(test_X_feats).float(), torch.from_numpy(np.array([0] * test_X_feats.shape[0])).long())
# %%
train_X_feats.shape
# %%
class MLP(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(train_X_feats.shape[1], 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=64, out_features=64, bias=False),
            # nn.Dropout(0.95),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(in_features=64, out_features=32, bias=False),
            # nn.Dropout(0.8),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Linear(32, len(class2idx))
        )
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        # f1 = f1_score(
        #     pd.Series(y_hat.detach().numpy()).astype('str'),
        #     pd.Series(y.detach().numpy()).astype('str'),
        #     average='micro'
        # )
        loss = self.ce(y_hat, y)
        self.log('train_loss', loss)
        # self.log('train_f1', f1)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.layers(x)
        loss = self.ce(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-3)
        return optimizer

# %%

mlp = MLP()
# max_epochs = 15
max_epochs = 5

early_stop_callback = EarlyStopping(
   monitor='val_loss',
   min_delta=0.00,
   patience=5,
   verbose=False,
   mode='min'
)

trainer = pl.Trainer(
    auto_scale_batch_size='power', 
    gpus=0, 
    deterministic=True, 
    callbacks=[early_stop_callback],
    max_epochs=max_epochs)

# %%
trainer.fit(
    mlp,
    DataLoader(dataset=train_dataset, batch_size=256), 
    val_dataloaders=DataLoader(dataset=val_dataset, batch_size=256)
)


# %%

def model_predict(mdl, dataset):

    y_pred_list = []
    with torch.no_grad():
        mdl.eval()
        for X_batch, _ in DataLoader(dataset):
            X_batch = X_batch.to('cpu')
            y_test_pred = mdl(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim=1)
            y_pred_list.append(y_pred_tags.cpu().numpy())

    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    return y_pred_list


y_pred_list = model_predict(mlp, val_dataset)

# %%
idx2class = {v: k for k, v in class2idx.items()}

# %%

val_preds_raw = pd.Series(y_pred_list).replace(idx2class)
val_preds = pd.Series(y_pred_list).replace(idx2class)
val_y_raw = pd.Series(val_y).replace(idx2class)


# %%
val_preds[val.code_pred.notnull()] = val.code_pred[val.code_pred.notnull()]
val_preds = val_preds.astype("int").astype("str")
print(f1_score(val_preds, val_y_raw, average='micro'))
print(f1_score(val_preds_raw, val_y_raw, average='micro'))
# print(f1_score(lr_cp.predict(train_X_feats), train_y, average='micro'))



# %%
if submit:
    mlp = MLP()
    max_epochs = 6
    trainer = pl.Trainer(auto_scale_batch_size='power', gpus=0, deterministic=True, max_epochs=max_epochs)
    trainer.fit(mlp, DataLoader(dataset=full_dataset, batch_size=256))


# %%
test_predictions = model_predict(mlp, test_dataset)
test_predictions = pd.Series(test_predictions).replace(idx2class)
test_predictions[test.code.notnull()] = test.code[test.code.notnull()]
# %%
test_predictions
# %%
test["ID"]
# %%
submission = test.copy().loc[:, ["ID"]]
# %%
submission["Codi QdC"] = test_predictions.astype("int")
submission["Codi QdC"]
# %%
if submit:
    submission.to_csv("submissions/08_first_nn.csv", index=False)

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
