---
title: "favourite pet?"
date: 2021-10-13T22:07:37+02:00
draft: false
categories: ["projects"]
---


## 1. What am I doing?

I'm trying to find out if you can tell if a person is talking about their *Python* or *dog*, when the person does not use any of the words "python" or "dog".

Apart from being funny for me, I want to practice creating and learning RNNs, LSTMs in particular.

## 2. Dataset

is taken from reddit, because of its simple API. This time [praw](https://praw.readthedocs.io/en/stable/), which is an excellent python package for downloading any information/data from reddit was not enough, as it allows to get only 1000 posts at a time. For this project I needed more, so I used [psaw](https://psaw.readthedocs.io/en/latest/), which may work as a thin wrapper around `praw` and does not have such constraint.

Here's the code that I used to download the data:

```{python}
import praw
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from psaw import PushshiftAPI


# reading reddit credentials
reddit_creds_path = Path.home() / "cookbook" / "scratchpad" / "rnn" / "creds.json"
with open(reddit_creds_path, "r") as f:
    reddit_creds = json.load(f)

# setting up a connection
reddit = praw.Reddit(
    client_id=reddit_creds["client_id"],
    client_secret=reddit_creds["client_secret"],
    user_agent="python_or_dog",
    username=reddit_creds["username"],
    password=reddit_creds["password"],
)
api = PushshiftAPI(reddit)

# a function for downloading posts
def get_data(api, subreddit_name):
    submissions = api.search_submissions(limit=10000, subreddit=subreddit_name)
    bodies = []
    for submission in tqdm(submissions, total=10000):
        bodies.append(submission.selftext)
    topics_data = pd.DataFrame(dict(body=bodies, label=subreddit_name))
    return topics_data


# downloading posts
all_posts = dict()
for pet in ["dogs", "Python"]:
    raw_data = get_data(api, subreddit_name=pet)
    all_posts[pet] = raw_data[raw_data["body"].str.len() > 200]
    print("downloading {} finished".format(pet))
    del raw_data  # save some memory

# saving results
result = pd.concat(all_posts.values())
save_path = Path.home() / "cookbook" / "scratchpad" / "rnn"
result.to_csv(save_path / "posts_reddit.csv", index=False)
```

I've decided to filter out those posts, which were shorter than 200 characters, as they probably do not convey enough information for the neural network.

## 3. Network

### Why keras, not pytorch?

This time I used [keras](https://greysweater42.github.io/keras) for creating and learning an LSTM network. I tried to use pytorch as well, but the library for working with text in pytorch is called [torchtext](https://pytorch.org/text/stable/index.html) and is nowhere near as good as other pytorch add-ons like torchvision, for several reasons:

- if you deal with a huge amount of data, which is often the case, you have to write your own generator as a sublass of data loader (which is not very difficult, but works exactly the same for every project, so could be provided by the package), while torchvision already has it;

- the API for datasets which fit into memory, e.g. stored in csv files, has a rather non-intuitive syntax: you have to define each column of the csv file and provide it to the data reader (`fields` argument):

```{python}
# dataset
LABEL = data.LabelField()
POST = data.Field(tokenize="spacy", lower=True, tokenizer_language="en_core_web_sm")
fields = [("body", POST), ("label", LABEL)]
dataset = data.TabularDataset(path="pytorch_data.csv", format="CSV", fields=fields)
train, test = dataset.split(split_ratio=[0.8, 0.2])
```

where it is not obvious how the data is stored. `torchtext` for each observation uses an object of class Example, which consists of two (or more) objects of class Example: text and label, which I find rather counter-intuituve. For my purpose, which is classification, storing text and label as instances of the same class seems wrong as they are of different types: *text* and *qualitative* (multinomial), in this case: *binary*.

Beside that, `keras`'s training is much more concise:

```{python}
history = model.fit(train, epochs=3, validation_data=test, validation_steps=30)
```

comparing to `pytorch`'s:

```{python}
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
```

and both of the training methods above do exactly the same... well, actually `model.fit` does even more. In this case pytorch seems to be a little bit too low-level, so maybe a good idea would be to use [fastai](https://greysweater42.github.io/fastai/), but still keras is a much more mature framework than fastai, which lacks in-depth documentation, various tutorials, wide community...

### Preparing data

The easiest way to read any amount of text data into a `keras` network is to use `text_dataset_from_directory`, but the data must be stored in a specific format:
```
.
├── train
│   ├── dog
│   │   ├── 1.txt
│   │   ├── 2.txt
│   │   └── ...
│   └── Python
│       ├── 1.txt
│       ├── 2.txt
│       └── ...
└── test
    ├── dog
    │   ├── 1.txt
    │   ├── 2.txt
    │   └── ...
    └── Python
        ├── 1.txt
        ├── 2.txt
        └── ...
```

The script below transforms csv file into the format above.

```{python}
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

DATA_PATH = Path("data")


def main():
    data = pd.read_csv("posts_reddit.csv")
    data["body"] = data["body"].str.replace("dog", "pet", case=False)
    data["body"] = data["body"].str.replace("Python", "pet", case=False)
    data["stage"] = np.random.choice(["train", "test"], len(data), p=[0.8, 0.2])

    nums = dict(train=defaultdict(lambda: 1), test=defaultdict(lambda: 1))
    for _, (body, label, stage) in data.iterrows():
        path = DATA_PATH / stage / label
        path.mkdir(exist_ok=True, parents=True)
        with open(path / f"{nums[stage][label]}.txt", "w") as f:
            f.write(body)
        nums[stage][label] += 1


if __name__ == "__main__":
    main()
```

When the data is ready, we can train the network (for working with GPU check out [my blog post on keras](https://greysweater42.github.io/keras/#gpu-support)):

```{python}
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import text_dataset_from_directory
from pathlib import Path

VOCAB_SIZE = 10000
DATA_PATH = Path("data")

train = text_dataset_from_directory(
    DATA_PATH / "train", labels="inferred", batch_size=32
)
test = text_dataset_from_directory(DATA_PATH / "test", labels="inferred", batch_size=32)

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train.map(lambda text, label: text))

model = tf.keras.Sequential(
    [
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=300,
            mask_zero=True,
        ),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1),
    ]
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)
history = model.fit(train, epochs=3, validation_data=test, validation_steps=30)
print("accuracy: ",  history.history['val_accuracy'][-1])

ls = []
y_hat = []
for texts, labels in test:
    y_hat.append(model.predict(texts).reshape(-1)) 
    ls.append(labels.numpy())

ls = np.concatenate(ls)
y_hat = np.concatenate(y_hat)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ls, y_hat > 0))
```
```
accuracy:  0.987500011920929
[[ 347    5]
 [  14 1185]]
```

I'm satisfied  with the resulting accuracy of 98.75%, but you can always challenge it using e.g. bidirectional or multilayer LSTM.
