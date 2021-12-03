---
title: "UNet"
date: 2021-11-27T15:01:58+01:00
draft: false
categories: ["Machine Learning"]
---

## 1. What is UNet?

[UNet](https://arxiv.org/abs/1505.04597) is a popular deep convolutional neural network used for object (image) segmentation.

## 2. How to learn UNet?

When I approach a new subject, I usually follow this path:

- Get a general idea of the subject. 

- Find a working implementation and run it. I found [it] pretty quickly.

- Build a working example on my own, which does not necessarily have to work perfectly, but it must have all the crucial functionalities to do their job.

- During the building process, before adding any of the functionalities, I do a quick recap if I understand what I am doing. If not, I deep dive into this particular feature and, e.g. write an article about it.

Following this path has several advantages:

- I can implement a decent model.

- I know how each of the functionalities of the model works.

- I can tweak/change any of these functionalities to adjust them to the specificity of my data.

Until recently I thought that understanding how something works requires building a replica by myself from ground-up, but this attitude has several disadvantages:

- I spend a lot of time on solving programming-specific problems, which do no lead to better understanding of the tool that I am trying to learn.

- In fact it does not concentrate on *understanding* at all, so I end up copy-pasting half-solutions from stackoverflow instead of *learning*, which is the main purpose.

- Ma main drive is almost always *curiosity*, and this approach doesn't support it. I get easily bored and frustrated, because there are virtually no checkpoints when I can admit "wow, I learned something, this is a great progress", and the final result is actually the code the I can easily find on the internet, which is even better and smarter and I should have used it instead of creating my own... Frankenstein of stackoverflow.


## 3. Learning

To get a general idea of image segmentation I watched a part of [cs231n from Stanford](https://www.youtube.com/watch?v=nDPWywWRIRo&t=676s). It gave me an overwiew of what I am going to do and why. 

After that I managed to find an excellent implementation of UNet on [github](https://github.com/milesial/Pytorch-UNet). It's excellence on a fact that I could download the code and run it, at least on an EC2 instance, as my laptop has only 2GB GPU. And it worked :)

After that I started to analyse the code. I needed to learn several things:

- [IoU and Dice coef](https://www.youtube.com/watch?v=AZr64OxshLo). Couriously, Dice coefficient seems to be exactly the same thing as F1 score and IoU only slightly varies from it.

- [Transposed convolution and upsampling](https://greysweater42.github.io/cnns/#6-transposed-convolution-and-upsampling), which seems to be specific not only to UNet, but to image segmentation in general.

- [AMP or Automatic Mixed Precision](https://towardsdatascience.com/understanding-mixed-precision-training-4b246679c7c4), which makes training lighter and faster (usually).

- [Schedulers in PyTorch](https://pytorch.org/docs/stable/optim.html) or how to automatically adjust the learning rate within epochs going.

And an implementation of a dumb net, which mimics the most important functionalities of UNet.

```{python}
import torch.nn as nn
import torch


class DumbNet(nn.Module):

    def __init__(self) -> None:
        super(DumbNet, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        x_down = self.down(x)
        x_up = self.up(x_down)
        x_out = self.out(x_up)
        return x_out
```

with down and upsampling.