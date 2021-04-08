---
title: "pytorch basics"
date: 2019-01-06T17:44:32+01:00
draft: false
categories: ["Python", "Machine learning"]
---

## 1. What is `pytorch` and why would you use it?

- pytorch is a python package which makes learning deep neaural networks relatively easy and fast

- it's main "rival" is tensorflow, as pytorch was released by Facebook, but tensorflow by Google

More on pytorch you will find in [this post](https://greysweater42.github.io/pytorch).

## 2. "Hello world" example

> inspired by [this article](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)

> using pytorch version x.xx

Let's define a simple quadratic function.

```{python, engine.path = '/usr/bin/python3'}
import numpy as np
import pandas as pd

theta = 2
x = np.random.rand(10) * 10
y = x ** theta
data = pd.DataFrame(dict(x=x, y=y))
```

`theta` is a parameter we'll be estimating. We will see how close to 2 our optimization algorithm gets us. Let's begin the estimation!

```{python, engine.path = '/usr/bin/python3'}
import torch
from torch.autograd import Variable

def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2).sum()))

def forward(x, e):
    """Forward pass for our fuction"""
    return x.pow(e.repeat(x.size(0)))

# initial settings
learning_rate = 0.00005

x = Variable(torch.FloatTensor(data['x']), requires_grad=False)
y = Variable(torch.FloatTensor(data['y']), requires_grad=False)

theta_hat = Variable(torch.FloatTensor([1]), requires_grad=True)

loss_history = []
theta_history = []

for i in range(0, 600):
    y_hat = forward(x, theta_hat)
    loss = rmse(y, y_hat)
    loss_history.append(loss.data.item())
    loss.backward()

    theta_hat.data -= learning_rate * theta_hat.grad.data
    theta_hat.grad.data.zero_()
    theta_history.append(theta_hat.data.item())
```

To get the final estimation we use the following command:

```{python, engine.path = '/usr/bin/python3'}
print(theta_hat.data)
```
