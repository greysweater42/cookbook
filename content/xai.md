---
title: "XAI"
date: 2020-11-22T12:38:18+01:00
draft: false
categories: ["Machine learning"]
---

# 1. Why even bother explaining machine learning models?

- users will trust the model which explains their decisions not only because of [this](https://leversofpersuasion.medium.com/because-to-persuade-give-a-reason-5f532f5b558a#:~:text=WHY%20GIVING%20A%20REASON%20WORKS&text=When%20you%20give%20someone%20a,%5D%20wanted%20to%20do%20anyway.%E2%80%9D)

- you will gain insight on whether the model will perform well on new data, e.g. you may find out that the decision if a photo depicts a husky or a dachshund is based not on features of a dog but on snow that lays all around.

# 2. How to do that?

The most popular approaches are:

### 1. [SHAP](https://arxiv.org/abs/1705.07874)

Useful resources:

- [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874) - a pretty good paper where the original SHAP values are defined, but definitely too complicated to begin with. You should read the three papers listed below. Having read them, you may get back to this one, from which you will learn that SHAP is pretty much Shapley value used in the context of linear regression and that calculating in directly is super ineffective, so some optimization techniques are used.

    - lime - as SHAP is a combination of Lime, SHapley value and 4 other methods (actually it is based very heavily on Shapley values, but used in a completely different context) - all the good resources I found are listed below.

    - [Analysis of regression in game theory approach](https://www.researchgate.net/publication/229728883_Analysis_of_Regression_in_Game_Theory_Approach) - where the idea of using Shapley values in regression appeared. There is also a word on why this seems to be a good idea (in linear regression we lack the statistics to measure the importance of *groups* of coefficients, so we completely forget about "synergies", or "cooperation/coalition", or "multicollinearity"). Permutations are explained in a rather straightforward way, but if you still could not grasp them, then try the next paper.

    - [Introduction to the Shapley value](http://www.library.fa.ru/files/roth2.pdf) - if you have economic background, you will feel at home, as the Shapley the Shapley value is explained here as the average marginal contribution to the score (which is e.g. R squared), and the average is weighted based on the probability that this particular group of coefficients appear together in the model.

- Reading only the paper above (and its explanatory papers) may be not enough to actually understand how SHAP works, but you will gain a broad understanding of the subject, e.g. where the whole idea came from. To go further, you can read [this chapter from Interpretable machine learning](https://christophm.github.io/interpretable-ml-book/shap.html)

### 2. [lime](https://arxiv.org/abs/1602.04938)

We pretty much build a linear regression model on predictions. Useful resources:

- [the official paper](https://arxiv.org/abs/1602.04938), where the method is presented in mathematical context

- [python lime package](https://github.com/marcotcr/lime) and its [docs](https://lime-ml.readthedocs.io/en/latest/index.html). Unfortunately after reading docs and github examples I still was not sure how I should interpret the results presented on plots. Anyway it is a good place to start getting familiar with python's lime syntax.

- [a chapter fromm Explanatory Model Analysis by pbiecek](https://pbiecek.github.io/ema/LIME.html) is the place where I finally understood what lime's plots mean. Even though examples are written in R, the plots are analogical to those produced by lime in python, and there is a mathematical explanation for what you can see in them.

- [Explaining the explainer: A First Theoretical Analysis of LIME](https://arxiv.org/abs/2001.03447) - an excellent, in-depth analysis oh how lime works on a simple example of linear regression model. Very useful to understand how lime actually works on tabular data and why you shouldn't trust its estimation values! (in a nutshell: results vary significantly depending on your choice of hyperparameter *v*, i.e. how much "local" you are, e.g. in Poland living near your parents means within 30km range, in Russia - probably around 300km ;) )

In general I it was extremely hard to find proper explanation of what actually lime does. Ironically, the "explainer" which is meant to transform a complex, nonlinear model into an easy to grasp form is very complicated and non-intuitive itself. The cases which I found particularly obscure are:

- when we calculate the distance between observations we use Gaussian Radial Basis Function (known also as Gaussian kernel - Zaki, Meira, Data Mining ana Analysis, p. 147), we have to choose arbitrarily the value of `v`. Depending on our choice, the results may vary (how much? - I will address this question below)

- our linear model may turn out to be quite poor. In this case we should not use this method at all. Shouldn't there be a warning somewhere?

- the coefficients of the explainer are rather unexplainable, because 

    - they depend on how good the linear model fits to the data

    - and on our prior choice of `v`

- subject that is not mentioned often: xgboost manages multicollinearity between variables pretty well, but linear regression unfortunately not. Its coefficients are not stable, so their interpretation is... risky.

To sum up, despite my concerns, I like this method. I think it is a brilliant idea to explain the black box models, but we should not be as yolo-optimistic as the authors of the majority of articles on medium.com, towardsdatascience.com etc. are and not treat lime as a magical tool that finally solves the black-box explainability problem. Actually I am quite disappointed that the hype on data science leads to dishonest psuedo-papers and false promises ("with lime you can explain any black-box model, because it's model agnostic" - rly? although it is true that it is model agnostic, the quality of the explanation may be very poor, even false and misleading if you do not do it carefully; besides in some cases it may not be possible to find a feasible linear approximation. Well, there are some '24-hour courses' on data science, which may suggest that after 24 hours... you are a competent data scientist? This is a subject for another discussion ;) )

### 3. [eli5](https://eli5.readthedocs.io/en/latest/)

TODO

### 4. CAM - Class Activation Mapping

In contrast to the previous methods, which are model agnostic, CAM works specifically for convolutional neural networks trained on images. Similarily to shap and lime, it highlights these areas on a given image, which had the biggest impact on prediction, e.g. in case of predicting a cat, it would highlight cat's mouth, ears and tail.

The concept of CAM is rather straightforward and intuitive: we calculate a weighted average of the channels in the last convolutional layers. Weight are provided by the network itself: the come from the linear layer which follows the convolutional layer. In result we recognize which channels from the last convolutional layer contributed the prediction the most, and if channel concentrated on a specific area of the image, which is often the case, we know which areas were the most important. The whole concept was presented in the article [Learning Deep Features for Discriminative Localization](https://arxiv.org/abs/1512.04150).

Unfortunately I haven't come across any decent implementation of CAM in pytorch (maybe there is one, but I couldn't find it). I've seen examples of CAM in keras, but I don't use it on a daily basis, and retraining a model in a different framework sounds like the last thing anyone would want to do. So here's a short script I wrote with the help of [Deep Learning for Coders with fastai and PyTorch](https://www.amazon.com/Deep-Learning-Coders-fastai-PyTorch/dp/1492045527):

```{python}
class Hook:
    def hook_func(self, m, i, o): self.stored = o.detach().clone()

hook_output = Hook()
hook = learn.model[0].register_forward_hook(hook_output.hook_func)  # 1

from pathlib import Path
import matplotlib.pyplot as plt
import torch

def cam(i):
    path = Path("data/fiat-126").ls()[i]  # 2
    img = PILImage.create(path)
    x, = first(loader.test_dl([img]))

    with torch.no_grad(): output = learn.model.eval()(x)
    act = hook_output.stored[0]
    cam_map = torch.einsum('ck,kij->cij', learn.model[1][-1].weight, act)

    x_dec = TensorImage(loader.train.decode((x,))[0][0])

    _, ax = plt.subplots()
    x_dec.show(ctx=ax)
    ax.imshow(cam_map[1].detach(), alpha=0.5, extent=(0,224,224,0), interpolation='bilinear', cmap='magma')

cam(0)
```

Two lines in this code may be difficult to grasp should you not know the context: 

- #1 `learn` is a fastai `Learner` instance, but instead od `learn.model` you can use any pytorch neural network you want. If you have never seen a pytorch hook: they are not that difficult, unfortunately I could not find any simple tutorial on them. It took quite a while to understand how they work. Anyway, they are not crucial to understand CAM.

- #2 in this example I gave the path to a file stored in "data/fiat-126", but you can (and should) provide a path to a picture that you want to explain.

> At first I was quite disappointed with how CAM works, or, actually, doesn't work. But maybe it just explained to me that my neural network made no sense, which is what I am still investigating. Despite the failure, I still believe in this method, as it seems it should work.

### 5. CNN layers

TODO