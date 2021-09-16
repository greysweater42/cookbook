---
title: "PCA / SVD"
date: 2021-08-31T18:32:03+02:00
draft: false
categories: ["Machine learning"]
---

## 1. What ic PCA and why do we care?

PCA stands for Principal Component Analysis and is a clever linear transformation, which usually serves the following purposes:

- dimenstionality reduction (in case of tabular data and text processing)

- noise reduction (in image and signal preprocessing)

It is worth mentioning that SVD (Singular Value Decomposition) is a generalization of PCA, so in practice these are exactly the same methods.

## 2. Why an article about a simple linear transformation?

Mainly because PCA is usually treated as a black box, while in fact it shouldn't. Many machine algorithms actually are black boxes (gradient boosting, neural networks etc.), but many of them aren't. IMHO, we shouldn't ever treat explainable algorithms as black boxes, i.e. we should always make an effort to understand them, because this leads to better models/analyses.
> I think we should *always* spend most of our time on *understanding* how the model/analysis works to gain the intuition, for machine learning is not only feeding any algorithm with the data and checking for performance. 

Besides, I personally find PCA (and SVD) interesting, as it very broadly used in all kinds of data: images, text, signals (time series) and measurements (I've never worked on videos by now, but this is a rather rare type of data, to my experience).

## 3. How does it work?

There are many resources that explain it better than I possibly could. My favourites are:

- materials for CS168 from Stanford: https://web.stanford.edu/class/cs168/l/l7.pdf and https://web.stanford.edu/class/cs168/l/l8.pdf - a good place to start with.

- [A tutorial on Principal Component Analysis by Jonathon Shlens](https://arxiv.org/pdf/1404.1100.pdf) - for me it was very useful, as the mathematical equations presented in this article closely resemble those in my favorite linear algebra textbook, and probably - in most linear algebra textbooks. After gaining general intuituin on the subject with the lectures from Stanford, now it is time for making a bridge between PCA and linear algebra.

- Data Mining and Analysis, Zaki, Meira - best book on data mining I've ever read. Chapter 7: "Dimensionality Reduction".

But the articles above do not mention programming at all (except the second one, which provides examples in Matlab). Here's a little code which shows how PCA can be used:

```{python}
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from sklearn.decomposition import PCA

k = 1  # number of target dimensions

# example data
# this data already has 0-mean
x = np.random.normal(0, 2, 5000)
y = 2 * x + np.random.normal(0, 1, 5000) 
X = np.array([x, y]).T  # a matrix with two columns: x and y

# using sklearn
from sklearn.decomposition import PCA
pca = PCA(1)  # we will reduce the number of dimensions from 2 to 1
pca.fit(X)
pca.singular_values_  # biggest eigenvalue of X.T @ X
pca.components_  # eigenvector of X.T @ X for biggest eigenvalue
pca.explained_variance_ratio_  # how much variance is explained by this eigenvector
Y = pca.transform(X)  # actual dimesionality reduction
X_hat = pca.inverse_transform(Y)  # back to 2 dimensions, used in case of noise removal

# using np.linalg, more low-level
eig_values, eig_vectors = np.linalg.eig(X.T @ X)
eig_order = eig_values.argsort()[::-1]  # descending
# according to docs: each column is an eigenvector, hence transposition; it's easier to 
# work on rows
eig_vectors = eig_vectors.T[eig_order][:k]
eig_vectors.shape
Y = X @ eig_vectors.T  # actual dimesionality reduction
X_hat = Y @ eig_vectors  # back to 2 dimensions, after removing noise
```

As can be seen in the code above, the number of dimensions to which data is reduced is set to `k = 1`, which is arbitrary in this case. In general we look at `pca.explained_variance_ratio` to see how much each of the new dimensions represents the data. Usually we are interested in those dimensions, for which cumulated variance ratio is higher than 90-95%, but there is no strict rule for that. We want to maximize two contradictory measures, i.e. minimize the number of dimensions and maximize variance, and there is no obvious answer to which of them is more important.

## 4. Examples

### tabular measurements - dimensionality reduction (most typical case)

Let's use `iris` in this example. The data consists of four types of measurements: petal length and width and sepal length and width. We may suppose that length and width are highly correlated, and we believe we could do with 2 generalized dimesions instead of 4: petal size and sepal size. Let's reduce the dimensions of iris to 2:

```{python}
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

X = load_iris()['data']
pca = PCA(4)
pca.fit(X)
print(np.cumsum(pca.explained_variance_ratio_))
```
```
array([0.92461872, 0.97768521, 0.99478782, 1.        ])
```

It seems that the first 2 dimensions will explain almost 98% of the variance of the dataset, that's pretty good. Let's interpret the eigenvectors:

```{python}
import pandas as pd
print(pd.DataFrame(pca.components_, columns=load_iris()['feature_names']))
```

```
    sepal length (cm)   sepal width (cm)    petal length (cm)   petal width (cm)
0    0.361387           -0.084523            0.856671            0.358289
1    0.656589            0.730161           -0.173373           -0.075481
2   -0.582030            0.597911            0.076236            0.545831
3   -0.315487            0.319723            0.479839           -0.753657
```

The first eigenvector `[0.361387	-0.084523	0.856671	0.358289]` concentrates mostly on petal length, the second one on sepal length and width, so our assumption that we can represent the data in two dimensions: sepal and petal size was right. (The only surpriging thing in the first eigenvector is a high value for sepal length, hence the assumption was rather "quite right".)

### images

[This](https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html#PCA-as-Noise-Filtering) is a perfect example of denoising images.
I found this book very interesting because (besides the algorithmic content) the author has a fantastic knowledge of numpy.

To be honest, I am not satisfied with how PCA denoises images and I would rather use a different method for this problem.

### text (LSI - Latent Semantic Indexing)

I wrote a [separate article](https://greysweater42.github.io/nlp/#exploration) about LDA (Latent Dirichlet Allocation) and how PCA is used for this purpose.

### signals

In case of brainwaves I recommend having a look at the [mne](https://mne.tools/stable/auto_examples/decoding/decoding_unsupervised_spatial_filter.html#analysis-of-evoked-response-using-ica-and-pca-reduction-techniques) package.

## 5. Afterthoughts

- eigenvalues and eigenvectors can be interpreted as Lagrange multipliers of a function `f(x) = x.T @ SIGMA @ x` with a constraint `x.T @ x = 1`, which is an excellent job interviewing questions if someone says PCA is "naah, easy" (SIGMA is a variance-convariance matrix). And how would you describe a variance-covariance matrix in the context of linear transformations? ;)