
# %%
import torch
import pandas as pd
import numpy as np

from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("codegram/calbert-base-uncased")
model = AutoModel.from_pretrained("codegram/calbert-base-uncased")
# %%

model_class = BertForSequenceClassification.from_pretrained("codegram/calbert-base-uncased")
# %%
train = pd.read_csv("data/train.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
test = pd.read_csv("data/test.csv").rename(columns={"TÍTOL": "title", "Codi QdC": "code"})
categories = pd.read_csv("data/categories.csv").drop(columns="Unnamed: 0").rename(columns={"Codi QdC": "code"})

# %%
# 1-hot encode and add special starting and end tokens
encoded_sentence = tokenizer.encode(["M'és una mica igual"] * 2)
print(encoded_sentence)
encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
embeddings = model(encoded_sentence)
embeddings["last_hidden_state"].detach().shape
# %%

encoding = tokenizer(list(train.title), return_tensors='pt', padding=True, truncation=True)

# %%
input_ids = encoding['input_ids']
attention_mask = encoding['attention_mask']
# encoded_titles = [tokenizer.encode(title) for title in train.title]
# encoded_titles = [np.array(tokenizer.encode(title)) for title in train.title]

# %%
labels = torch.tensor(train.code).unsqueeze(0)
# %%
# outputs = model_class(input_ids, attention_mask=attention_mask, labels=labels)
# loss = outputs.loss
# loss.backward()
# optimizer.step()
# %%
outputs = model_class(
    input_ids[:10,:],
    attention_mask=input_ids[:10,:],
    labels=torch.tensor(train.code[:10]).unsqueeze(0)
)
loss = outputs.loss
loss.backward()
optimizer.step()

# %%
train.code[:10]
# %%
