# pip install spacy
# python -m spacy download en_core_web_sm

from torchtext.legacy import data
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix

from net import Net


device = "cuda"

# dataset
LABEL = data.LabelField()
POST = data.Field(tokenize="spacy", lower=True)
fields = [("body", POST), ("label", LABEL)]
dataset = data.TabularDataset(
    path="data.csv", format="CSV", fields=fields, skip_header=False
)
train, test = dataset.split(split_ratio=[0.8, 0.2])

# vocabulary
vocab_size = 10000
POST.build_vocab(train, max_size=vocab_size)
LABEL.build_vocab(train)  # fixes weird error

# data loaders
train_iterator, test_iterator = data.BucketIterator.splits(
    (train, test),
    batch_size=32,
    device=device,
    sort_key=lambda x: x.body,  # fixes weird error
    sort_within_batch=True,  # fixes weird error
)

# training neural network
model = Net(100, 300, vocab_size + 2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

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
