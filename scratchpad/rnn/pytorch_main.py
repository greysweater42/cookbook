# pip install spacy
# python -m spacy download en_core_web_sm

from torchtext.legacy import data
from tqdm import tqdm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix


device = "cpu"

# dataset
LABEL = data.LabelField()
POST = data.Field(tokenize="spacy", lower=True, tokenizer_language="en_core_web_sm")
fields = [("body", POST), ("label", LABEL)]
dataset = data.TabularDataset(path="pytorch_data.csv", format="CSV", fields=fields)
train, test = dataset.split(split_ratio=[0.8, 0.2])

# vocabulary
POST.build_vocab(train, max_size=10000) # , vectors = 'glove.6B.200d')
LABEL.build_vocab(train)  # fixes `"LabelField" has no attribute "vocab"`

# data loaders
train_iterator, test_iterator = data.BucketIterator.splits(
    (train, test),
    batch_size=32,
    device=device,
    sort_key=lambda x: x.body,  # fixes weird error
    sort_within_batch=True,  # fixes weird error
)

# neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(len(POST.vocab), 300)
        self.encoder = nn.LSTM(input_size=300, hidden_size=100, num_layers=1)
        self.predictor = nn.Linear(100, 1)

    def forward(self, seq):
        _, (hidden, _) = self.encoder(self.embedding(seq))
        preds = self.predictor(hidden.squeeze(0))
        return preds


# training neural network
model = Net()
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.BCEWithLogitsLoss()
# criterion = torch.nn.MSELoss()
# criterion = torch.nn.BCELoss()

epochs = 10
for epoch in range(1, epochs + 1):
    # training
    model.train()
    training_loss = 0.0
    for batch in tqdm(train_iterator):
        optimizer.zero_grad()
        predict = model(batch.body)
        loss = criterion(predict.squeeze(), batch.label.to(torch.float32))
        loss.backward()
        optimizer.step()
        training_loss += loss.data.item() * batch.body.size(0)

    # evaluation on test set
    model.eval()
    test_loss = 0.0
    for batch in tqdm(test_iterator):
        predict = model(batch.body)
        loss = criterion(predict.squeeze(), batch.label.to(torch.float32))
        test_loss += loss.data.item() * batch.body.size(0)
    print(f"Epoch: {epoch}, training loss: {training_loss}, test loss: {test_loss}")


# metrics
preds = []
labels = []
for batch in tqdm(test_iterator):
    predict = model(batch.body)
    preds.append(predict.detach().cpu())
    labels.append(batch.label.cpu())

preds = torch.cat(preds)
labels = torch.cat(labels)

print(accuracy_score(preds > 0.5, labels))
print(confusion_matrix(preds > 0.5, labels))

import numpy as np

np.unique(preds.numpy().reshape(-1), return_counts=True)
np.unique(labels.numpy().reshape(-1), return_counts=True)
