---
title: "CNNs"
date: 2021-11-05T10:16:34+01:00
draft: false
categories: ["Machine learning"]
---

## 1. What are CNNs and why would you care?

CNN stands for convolutional neural network, which is a type of artificial neural network used primarily for image classification, detection and segmentation.

CNN also means Cable News Network, or one of the most popular TV networks in US, but who cares.

## 2. But what is a [convolution](https://en.wikipedia.org/wiki/Convolution)?

A convolution is a measure of similarity between two functions *f(t)* and *g(t)* depending on the moment t, so it actually is another function, *h(t)*. For continues functions it is defined as an integral over product of all the possible shifts between two functions (shift being usually denoted as $\tau$), hence the definition:

$$ (f * g)(t) = \int f(\tau) g(t-\tau) d\tau $$

If you don't feel confident with this formula, there are plenty of tutorials on youtube, like [this one](https://www.youtube.com/watch?v=N-zd-T17uiE), but to get the full picture I recommend familiarizing yourself with digital signal processing, e.g. Fourier transform or wavelet transform, to get a feeling of how and why signals are processed.

But if you don't feel comfortable with integrals, you may have a look at the discrete case:

$$ (f*g)[n] = \sum_{m=-\infty}^{\infty} f[m]g[n-m] $$

where you can clearly see that a convolution is a function, which is a combination of two functions, and has high values if the two functions are "alike" at specific moment of time.

For more information you can refer to the beginning of Ian Goodfellow's *Deep Learning*, chapter 9: Convolutional Networks.

## 3. Simple examples of a convolution

### a) implemented "by hand"

Let's define some data:
```{python}
import numpy as np

f = np.zeros(12)
f[8] = 1
f[9] = -1

g = np.array([0., 1., -1., 0.])
```

We defined two simple functions: *f* ang *g*, which look exaclty the same at some point. The second one is much shorter though, but this is not a problem for convolution: it simply assumes that the other values of the function are zeroes.

Let's apply a convolution to these functions:
```{python}
s = max(len(f), len(g))
x = np.concatenate([np.zeros(s), f, np.zeros(30)])
y = np.concatenate([np.zeros(s), g, np.zeros(len(f) - len(g)), np.zeros(s)])

c = np.zeros(len(f) + len(g) - 1)
for t in range(len(f)):
    for tau in range(-s, s):
        c[t] += x[s+tau] * y[s+t-tau]

print(c)
```
```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1. -2.  1.  0.  0.  0.]
```
First, we declared a margin called `s`, which is just for convenience of numpy during shifting. In a loop we multiply specific values of the function, just as in the discrete convolution case, but this time I used variable names for continues convolution: t and tau, instead of n and m.

The convolution has length 15, which is a sum of lengths of domains of f and g, minus 1 (the number of possible combinations, when *f* and *g* are in contact).

### b) using numpy

Couldn't be easier:

```{python}
print(np.convolve(f, g))
```
```
[ 0.  0.  0.  0.  0.  0.  0.  0.  0.  1. -2.  1.  0.  0.  0.]
```

Same result as before.

### c) using pytorch

Pytorch, being a framework for artifical neural networks, has a function called *convolution*. Let's see how it works:

```{python}
import torch

f = torch.zeros(12)
f[8] = 1
f[9] = -1
g = torch.tensor([0., 1., -1., 0.])

conv = torch.nn.Conv1d(1, 1, (1, 4), bias=False)
conv.weight = torch.nn.Parameter(g.unsqueeze(0).unsqueeze(0).unsqueeze(0))
# batch_size = 1, "image" "depth": 1 (not RGB), "image" size: 20x1
f = f.unsqueeze(0).unsqueeze(0).unsqueeze(0)
print(conv(f).detach()[0][0][0])
```
```
tensor([ 0.,  0.,  0.,  0.,  0.,  0., -1.,  2., -1.])
```

In this case we had to `unsqueeze` (add a dimension) to our function/array several times, so it was perceived as an image of size 20x1, depth 1 and belonging to a batch of size 1, because you use all of these parameters for training a NN and they are the defaults for pytorch.

Curiously, the results are slighlty different for pytorch, comparing to e.g. numpy: it turns out that pytorch uses cross-correlation instead of convolution, which in fact makes no difference, as they are almost exaclty the same transformations ([just have a look at Wikipedia](https://en.wikipedia.org/wiki/Cross-correlation#Properties)). A small discussion about this you may find also on [stackoverflow](https://stackoverflow.com/questions/66640802/why-are-pytorch-convolutions-implemented-as-cross-correlations).

## 4. Differences between convolutional and fully-connected layers

The answer I would have given before deep diving into convolutions:

>Well, these are just completely different things.

But now, as I got smarter ;), I know that they are surprisingly similar. At first sight the differences are:

>Fully-connected layer takes as input a *n*-dimensional vector, but a convolutional layer a *x* by *y* by *d* tensor.

This does not have to be the case. Convolutional layer *can* take a *x* by *y* by *d* tensor as input, but can also take a *n*-dimmensional vector. Actually it is widely used for image classification thanks to its shape-awareness.

>Fully-connected layer produces a *n*-dimensional vector, while convolutional layer a *a* by *b* by *k* tensor (where a ~= (x + padding) / stride etc.) and *k* is the number of kernels.

This also does not have to be case. You can set the number of kernels to 1, which would immediately make it more similar to a fully-connected layer and if your input is an *n*-dimensional vector, the output would also be a *n*-dimensional vector (it might be *n+/-k*, depending on specification).

>There are completely different algorithms to produce an output: convolution and matrix multiplication.

*Completely* is a subjective term, they just are different, yet they share some similarities. In fact we could understand convolution as a specific, selective and iterative kind of matrix multiplication. There are other differences, but IMHO considering a convolution as a remote relative of matrix multiplication is an interesting point of view, which is worth exploring.

## 5. Quirks in CNNs

- stride and padding

- pooling

- activation - this actually works the same

- XAI works quite well, especially [grad-cam](https://greysweater42.github.io/fiat_ferrari/#6-but-does-it-really-work)

- regularization - data augmentation and dropout (should dropout be used for CNNs?)

## 6. Transposed convolution and upsampling

If you work on image segmentation, you surely come across the idea of "upsampling* or *transposed convolution*. In short, image segmentation relies on a concept of encoding (downsampling) an image into a lower dimension and bringing it back to the initial size with decoding (upsampling) to predict a mask (an image where each pixel belongs to a class, usually reflected in color). Downsampling is usually done with *pooling* or convolution with *stride*. Both of these methods have the same effect: they chose every second (for stride=2 or maxpool_size=2) pixel from feature map.

Let's have a look at a modest prove (proving by showing an example is *not* a proving...) ok, let's show an example which *shows* that `stride` is the same think as taking every n-th element of a tensor:

```{python}
import torch
import torch.nn as nn

# unsqueeze(0) n times
u = lambda x, n=3: u(x.unsqueeze(0), n-1) if n > 0 else x

f = torch.zeros(12)
f[8] = 1
f[9] = -1
g = torch.tensor([0.0, 1.0, -1.0, 0.0])

kwargs = dict(in_channels=1, out_channels=1, kernel_size=(1, 4), bias=False)
conv1 = torch.nn.Conv1d(**kwargs)  # stride 1
conv1.weight = torch.nn.Parameter(u(g))
conv2 = torch.nn.Conv1d(**kwargs, stride=2)  # stride 2
conv2.weight = torch.nn.Parameter(u(g))

print(all(conv1(u(f))[0][0][0][::2] == conv2(u(f))[0][0][0]))  # the same
```
```
True
```

>I used a recurrent unsqueezer, as the only way I found to avoid writing `unsqueeze(0).unsqueeze(0).unsqueeze(0)`. Theoretically you can use `view()`, but it works slighlty differently for 2D tensors, so I tend to avoid it (similar to `resize` as a transformer in torchvision and `torch.reshape` - may seem like distant relatives, but are used in completely different contexts).

Now that we've recollected how downsampling works in CNNs, let's move back to upsampling. There are two popular ways to perform upsampling in pytorch. The easy way:

```{python}
print(conv2(u(f)))  # downsampled
upsample = torch.nn.Upsample(scale_factor=(1, 2))
print(upsample(conv2(u(f))))  # back to the previous dimensions
```
```
tensor([[[[ 0.,  0.,  0., -1., -1.]]]], grad_fn=<ThnnConv2DBackward0>)
tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0., -1., -1., -1., -1.]]]],
       grad_fn=<UpsampleNearest2DBackward1>)
```

The `Upsample` module can take various parameters depending on how you want the data to be upsampled. The default way presented above simply repeats the value `scale_factor` n times, just as numpy's `np.reapeat` function. More on that in [pytorch docs](https://pytorch.org/docs/stable/generated/torch.nn.Upsample.html).

But there is a smarter, learnable approach to upsampling: you can actually use a specific kernel just as if you were doing an operation inverse to convolution (which, mathematically speaking, is not strictly *inverse*).

```{python}
convtrans = torch.nn.ConvTranspose1d(**kwargs, stride=2)
convtrans.weight.data = u(g)
print(convtrans(conv2(u(f))).shape == u(f).shape)  # the same shapes after unpooling/upsampling
print(convtrans(conv2(u(f))))
```
```
True
tensor([[[[ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  1., -1.,  1.,  0.]]]],
       grad_fn=<SlowConvTranspose2DBackward0>)
```

As you can see, transposed convolution restores the initial size of the tensor, but the tensor has different values, so the operation is not fully *inverse* (probably there is a way to inverse the kernel properly, but haven't figured it out yet).
