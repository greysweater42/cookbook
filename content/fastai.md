---
title: "fastai"
date: 2021-03-03T18:26:08+01:00
draft: false
categories: ["Python", "Machine learning"]
---


## 1. What is fastai and why would you care?

It is a deep learning framework built on top of pytorch, which facilitates some frequently used tasks, e.g.:

- you can train your model in one line of code (and it works really well)

- you can easily view your dataset, which is particularly useful when working with images

- you can easily check where the model had the biggest errors.

## 2. A typical fastai program

### Dataset and Dataloader

Just as in [pytorch](https://greysweater42.github.io/pytorch), we begin with dataset and dataloaders. In case of images fastai provides an `ImageDataLoaders` abstraction, which consists of torch Dataset and DataLoader in one object. Unlike with torch.datasets.ImageFolder, you keep all your images in one folder. Besides, fastai lets us view some images with one simple command: `loader.show_batch`. The images shown are already transformed with `item_tfms` (transformation for validation dataset) and `batch_tfms` (transformation for validation and augmentation). fastai also provides us with a handy `aug_transforms` default augmentation function, which works surprisingly well out of the box.

```{python}
from fastai.vision.all import ImageDataLoaders, Resize, aug_transforms

loader = ImageDataLoaders.from_folder(
    path="data", 
    item_tfms=Resize(224), 
    batch_tfms=aug_transforms(), 
    valid_pct=0.2, 
    bs=20
)
loader.show_batch()
```

### Neural network, training

Probably the most commonly used practice for training deep neural networks is transfer learning, when we train our own network on top of a network already trained on a huge dataset (e.g. resnet34). fastai makes it surprisingly easy to apply this method. In this case we use `loader` defined in the previous step, instantiate a learner object and run a `fine_tune` method for 2 epochs. Why fine_tune? Because we merely update the neural network's weights starting with the weights provided by resnet34.
```
from fastai.vision.all import (
    cnn_learner,
    error_rate,
    resnet34
)

learn = cnn_learner(loader, resnet34, metrics=error_rate)
learn.fine_tune(2)
```

### Evaluation
Once we have trained our network, we check how it perfmorms on a validation dataset. We use a `ClassificationInterpretation` object for this purpose and it's invaluable `plot_confusion_matrix` and `plot_top_losses` methods, the names of which are rather self-explanatory.

```
from fastai.vision.all import ClassificationInterpretation

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(3, 3), dpi=80)
interp.plot_top_losses(5, nrows=2, ncols=3)
```

## 3. Customizing fastai with pytorch

In many cases we would rather use a different dataloader, neural network architecture or evaluation metrics, which are not provided by fastai. Overriding these functionalities causes several inconveniences. Lets have a look at some of them.

### Custom transformer, dataset or dataloader

Usually we would rather write our own augmentation transformer instead of using the default one provided by fastai, `aug_transforms`. To do this, we may learn the whole new bunch of transformers provided by fastai, but we may prefer to use those that we have been using so far, written in pytorch. fastai is built on top of pytorch, after all.
It turns out that in order to do that, you cannot simply replace your fastai transformers with those from torchvision.

````
# DOES NOT WORK - AN EXAMPLE CASE
from fastai.vision.all import ImageDataLoaders, aug_transforms
from torchvision import transforms

transforms = transforms.Compose([transforms.Resize(64, 64), ...])
loader = ImageDataLoaders.from_folder(..., batch_tfms=transforms, ...)
# DOES NOT WORK - AN EXAMPLE CASE
````

You have to create a pytorch DataSet instance, in this case an ImageFolder, and then instantiate a fastai dataloader using its `from_dsets` class factory method. Then, in order to be able to use ClassificationInterpretation, you should update `vocab` parameter in both `train` and `valid` datasets.

```
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import random_split
from fastai.vision.all import DataLoaders
from fastai.data.transforms import CategoryMap

IMAGE_SIZE = (224, 224)
trans = transforms.Compose({
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
})
data = ImageFolder("data", transform=trans)

train_len = int(len(data) * 0.8)
valid_len = len(data) - train_len
train, valid = random_split(data, lengths=[train_len, valid_len])
# for torch compatibility with ClassificationIntepretation
train.vocab, valid.vocab = CategoryMap(data.classes), CategoryMap(data.classes)  

dls = DataLoaders.from_dsets(train, valid, bs=20)
```

### Custom neural network architecture

Sometimes you may not want to use transfer learning or use it in a special way, e.g. freezing only specific layers. As fastai is built on top of pytorch, you can train your models on your own net architecture. In this case you define you own learner using a Learner object and train it with a `fit_one_cycle` method, which has a slightly misleading name: you provide the number of epochs (in the example below it is 10) and you can also choose the maximum learning rate, as after each epoch fastai chooses the best learning rate using its clever [learning rate finder](https://fastai1.fast.ai/callbacks.lr_finder.html). Unfortunately `plot_top_losses` no longer works.

```
from fastai.vision.all import Adam, nn, Learner, CrossEntropyLossFlat, accuracy

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        out_channels = 32
        max_pool = 2
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(max_pool, max_pool),
        )
        self.out_size = int(out_channels * IMAGE_SIZE[0] * IMAGE_SIZE[1] / (max_pool ** 2))
        self.fc = nn.Linear(self.out_size, 2)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x.view(-1, self.out_size))
        return x

learn = Learner(dls, Net(), loss_func=CrossEntropyLossFlat(), opt_func=Adam, metrics=[accuracy])
learn.fit_one_cycle(10, 0.001)
```

## 4. Interesting resources

- Most of the stuff about fastai I've learned from [Deep Learning for Coders with fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527). Unfortunately this book is targeted at beginners in programming and machine learning, so if you move from pytorch to fastai, you will end up skipping 95% of the content. But the remaining 5% is purely useful.

- You can also have a look at [fastai docs](https://docs.fast.ai/), which were not particularly useful for me. My issues turned out to be rather unusual (moving from pytorch to fastai).

- And there is also official [fastai's webpage](https://www.fast.ai/) with its mission clearly explained: *Making neural nets uncool again* ;)
