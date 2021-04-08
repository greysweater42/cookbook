---
title: "pytorch"
date: 2021-02-17T16:13:51+01:00
draft: false
categories: ["Python", "Machine learning"]
---

## 1. What is pytorch and why is it interesting?

* Pytorch is one of the most popular artificial neural network python packages, used mainly for deep learning.

* Comparing to other NN frameworks, pytorch:

    - makes using CUDA trivial

    - has excellent abstractions for working with neural networks (dataset, dataloader, nn.Module)

    - works well with Tensorboard

## 2. A typical NN structure

Pytorch NN scripts share a very similar structure: many models used in production environment are even almost exactly the same. Let's have a look at these common components.

### Dataset

Dataset class presents your data in a python-friendly manner. Having spent some time on processing data, you may have noticed that the most basic characteristics of a dataset are:

* its **length**, which means that a dataset actually *is* a *set* of rows/observations/items, 

* so you can **extract** any **subset** according to your wish

* and each dataset usually consists of raw, unprocessed data, where all the rows/observations/items should be **transformed** in exactly the same way right at the start.

Pytorch `Dataset` class complies with all the specifics above. Let's see a basic example:
```
from torch.utils.data import Dataset
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = data.loc[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample
```

Quite often you don't have to create your own Dataset class. You can use one of pytorch's predefined datasets, e.g.  `from dataset import Mnist` or classes written specifically for handling a particular type of data, e.g. [ImageFolder](https://pytorch.org/vision/0.8/datasets.html#imagefolder).

More on datasets you will find in [pytorch docs](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html).

### DataLoader

Once you've defined your dataset you can concentrate on how to feed the data into your neural network. Pytorch proposes using a handy `DataLoader` object:

```
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=10, shuffle=True)
```
which returns an iterator over your dataset. The arguments to the loader's initializer seem rather self-explanatory, except maybe one of them, which turns out to be particurarily useful: `sampler`, which can balance the dataset if necessary. More on `WeightedRandomSampler` you will find in [pytorch docs](https://pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler).

### Neural Net

We have finally reached the core part of the whole script: the neural network definition. Classically, as most of the neural networks use backpropagation for optimization, neural network performs two types of operation: **forward** and backward passes. Pytorch handles backward pass by itself as a reversed forward pass, however it is us who defines the forward pass, which in practice is equivalent to defining the whole neural network's structure.

An example of a very simple convolutional network:
```
import torch.nn as nn

IMAGE_SIZE = (64, 64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Linear(32 * IMAGE_SIZE[0] * IMAGE_SIZE[1], 2)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(-1, 32 * IMAGE_SIZE[0] * IMAGE_SIZE[1]))
        return x
```

In practice, you often use transfer learning, i.e. instead of learning all the weights by yourself, you load the weights provided by researchers or any external companies to the majority of layers of your network. Example:

```
import torch.nn as nn
from torchvision import models

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.ResNet50(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(self.model.fc.in_features, 500),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 2),
        )

    def forward(self, x):
        return self.model(x)
```

Models which you can use for transfer learning are briefly described in [pytorch docs](https://pytorch.org/vision/stable/models.html). A more in-depth tutorial you will find in [Programming PyTorch for Deep Learning](https://www.amazon.com/Programming-PyTorch-Deep-Learning-Applications/dp/1492045357), chapter 4.

### training the model

In this tutorial I do not use CUDA anywhere, as I try to keep the examples as simple as possible, but in real life you may be tempted to speed up the training process, despite the relatively high cost of renting a GPU. A minimal example of training a neural net:

```
net = Net()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for i in range(5):
    print(f"Epoch number: {i}")
    for idx, (inputs, labels) in enumerate(train_loader):
        if not idx % 5:
            print(idx)
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        optimizer.zero_grad()  # zero the previous gradients
        loss.backward()  # calculates the gradient on weights
        optimizer.step()  # goes down the gradient
```

### evaluating the model

Model evaluation remains a rather difficult subject and deep neural nets are still being considered black boxes, which means that they may behave in unexpected ways under specific circumstances.  In practice, I usually stick to the traditional machine learning techniques used for classification and regression models (ROC curve, precision/recall, accuracy etc.), but as a hobby I keep searching for more informative metrics provided by additional models or algorithms, which I briefly discuss in [this post on *explanation*](https://greysweater42@github.io/explanation).

A trivial example of accuracy measure and confusion matrix:
```
from sklearn.metrics import confusion_matrix

net.eval()

pos = 0.0
n_all = 0
trues = []
preds = []

for inputs, labels in test_loader:
    outputs = net(inputs)
    pos_loc = sum(torch.max(outputs, 1).indices == labels).item()
    pos += pos_loc
    n_all += len(inputs)
    preds.append(torch.max(outputs, 1).indices.numpy())
    trues.append(labels.numpy())
    print(pos_loc, len(inputs))

print(f"Accuracy: {pos / n_all}")

print("Confusion matrix:")
print(confusion_matrix(np.concatenate(trues), np.concatenate(preds)))
```

## 3. Recommended resources

- My favourite book on Pytorch is by now [Programming PyTorch for Deep Learning](https://www.amazon.com/Programming-PyTorch-Deep-Learning-Applications/dp/1492045357)

- and a legendary book by Ian Goodfellow: [Deep Learning (Adaptive Computation and Machine Learning series)](https://www.amazon.com/Deep-Learning-NONE-Ian-Goodfellow-ebook/dp/B01MRVFGX4), which concentrates mostly on mathematical side of neural networks and (I think) does not even mention Pytorch at all. Worth reading, in any case. Knowing pytorch is useless unless you understand neural networks well.

