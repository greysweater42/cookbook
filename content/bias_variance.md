---
title: "bias/variance trade-off"
date: 2022-01-15T17:49:33+01:00
draft: false
categories: ["Machine learning"]
---

## 1. Why an article about such a fairly straightforward subject as bias/variance trade-off?

IMHO, the subject is straightforward only superficially, so it may leave us with a false feeling of comprehension. From my statistician's point of view, there is a little more to think about.

## 2. When do we face this dillema?

**When we assess the quality of our model.**

To do that, we use [validation methods and metrics](https://greysweater42.github.io/validation). One of the basic rules is that we calculate the metrics on the **test set**. A commonly used metric for regression problems is MSE (Mean Squared Error), which corresponds to variance of the error (for many algorithms error by definition has mean 0).

MSE consists of 3 components:

- irreducible error, usually denoted as $\epsilon$ (in statistics) or $\xi$ (in machine learning), which is the noise that is far beyond the reach of the model, e.g. [measurement error](https://en.wikipedia.org/wiki/Observational_error), and we have no chance to predict it

- bias (squared), which is a systematic error of our model resulting from a wrong (e.g. too simple) function of the process. In literature a common example is using linear regression instead of a generalization of [logistic function](https://greysweater42.github.io/nls). In this case a basic model validity test, e.g. [White test](https://en.wikipedia.org/wiki/White_test) for heteroskedasticity would fail, but in general case we use more sophisticated/complex models than linear regression.

- variance, which results from the following assumptions:

    - our training dataset was perfectly (without biases and with perfect stratification) sampled from the population (which we are *not* sure of); it other words, it makes a *representative* sample of the population. This assumption is usually omitted in literature, as we assume that the exact subsample of the population that we work on was chosen deterministically. Or that as we can't measure it, this kind of error is irreducible.

    >I'm going to look closer at this assumption. We obviously have no data but our full dataset to prove the unrepresentativeness of our dataset, but *omitting* this subject does not mean it is negligible (this sort of thinking is a common logical fallacy). Thing gets interesting with bootstrap, even more with bootstrap with subsampling, when we actually *have* access to more observations.

    - the *noise* of the training dataset is minimal, so we can safely adjust our model to fit the exact values of the training data. 

    - in result, the model fits too tightly to the training dataset and does not generalize well to new data. In other words, the new dataset (e.g. test dataset) has **too much variance** for our model to understand.

In short:

$$ Error(x_0) = Irreducible Error + Bias^2 + Variance $$

from [The Elements of Statistical Learning, 2nd Edition by Hastie, Tibshirani, Friedman](), eq. 7.9, which slowly becomes my all-time ml/statistics textbook.


## 3. Connotations

- **High bias** results from too simplistic representation of the data. In other words, we can say the model has too low **complexity**, **flexibility** or too few **degrees of freedom**.

- Models with **low bias** are commonly called **underfitted**.

- **High variance** goes with **overfitting**. Overfitting results from high variance.

//TODO: connotations with learning curves
