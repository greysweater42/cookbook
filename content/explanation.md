---
title: "Explanation"
date: 2020-11-22T12:38:18+01:00
draft: false
categories: ["scratchpad"]
---

# 1. Why even bother explaining machine learning models?

- users will trust the model which explains their decisions not only because of [this](https://leversofpersuasion.medium.com/because-to-persuade-give-a-reason-5f532f5b558a#:~:text=WHY%20GIVING%20A%20REASON%20WORKS&text=When%20you%20give%20someone%20a,%5D%20wanted%20to%20do%20anyway.%E2%80%9D)

- you will gain insight on whether the model will perform well on new data, e.g. you may find out that the decision if a photo depicts a husky or a dachshund is based not on features of a dog but on snow that lays all around.

# 2. How to do that?

Most popular approaches are:

### 1. [SHAP](https://arxiv.org/abs/1705.07874)

- [Shapley's original paper from 1951](https://www.rand.org/content/dam/rand/pubs/research_memoranda/2008/RM670.pdf) - quite useless, actually.

TODO

### 2. [lime](https://arxiv.org/abs/1602.04938) - we pretty much build a linear regression model on predictions. Useful resources:


#### Useful resources:

- [the official paper](https://arxiv.org/abs/1602.04938), where the method is presented in mathematical context

- [python lime package](https://github.com/marcotcr/lime) and its [docs](https://lime-ml.readthedocs.io/en/latest/index.html). Unfortunately after reading docs and github examples I still was not sure how I should interpret the results presented on plots. Anyway it is a good place to start getting familiar with python's lime syntax.

- [a chapter fromm Explanatory Model Analysis by pbiecek](https://pbiecek.github.io/ema/LIME.html) is the place where I finally understood what lime's plots mean. Even though examples are wriiten in R, the plots are analogical to those produced by lime in python, and there is a mathematical explanation for what you can see in them.

- [Explaining the explainer: A First Theoretical Analysis of LIME](https://arxiv.org/abs/2001.03447) - an excellent, in-depth analysis oh how lime works on a simple example of linear regression model. Very useful to understand how lime actually works on tabular data and why you shouldn't trust its estimation values! (in a nutshell: results vary significantly depending on your choice of hyperparameter *v*, i.e. how much "local" you are, e.g. in Poland living near your parents means within 30km range, in Russia - probably around 300km ;) )

In general I it was extremely hard to find proper explanation of what actually lime does. Ironically, the "explainer" which is meant to transform a complex, nonlinear model into an easy to grasp form is very complicated and non-ituitive itself. The cases which I found particurarly obscure are:

- when we calculate the distance between observations we use Gaussian Radial Basis Function (known also as Gaussian kernel - Zaki, Meira, Data Mining ana Analysis, p. 147), we have to choose arbitrarily the value of `v`. Depending on our choice, the results may vary (how much? - I will address this question below)

- our linear model may turn out to be quite poor. In this case we should not use this method at all. Shouldn't there be a warning somewhere?

- the coefficients of the explainer are rather unexplainable, because 

    - they depend on how good the linear model fits to the data

    - and on our prior choice of `v`

- subject that is not mentioned often: xgboost manages multicollinearity between variables pretty well, but linear regression unfortunately not. Its coefficients are not stable, so their interpretation is... risky.

To sum up, despite my concerns, I like this method. I think it is a brilliant idea to explain the black box models, but we should not be as yolo-optimistic as the authors of the majority of articles on medium.com, towardsdatascience.com etc. are and not treat lime as a magical tool that finally solves the black-box explainability problem. Actually I am quite disappointed that the hype on data science leads to dishonest psuedo-papers and false promises ("with lime you can explain any black-box model, because it's model agnostic" - rly? although it is true that it is model agnostic, the quality of the explanation may be very poor, even false and misleading if you do not do it carefully; besides in some cases it may not be possible to find a feasible linear approximation. Well, there are some '24-hour courses' on data science, which may suggest that after 24 hours... you are a competent data scientist? This is a subject for another discussion ;) )

#### Examples:

TODO

#### Analysis:

TODO

### 3. [eli5](https://eli5.readthedocs.io/en/latest/)

TODO
