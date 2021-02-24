---
title: "basic machine learning algorithms"
date: 2019-04-22T18:05:21+02:00
draft: false
categories: ["Machine learning"]
tags: []
---

## 1. What is machine learning and why would you use it? 

* it's a rather complicated, yet beautiful tool for boldly going where no man has gone before.

* in other words, it enables you to extract valuable information from data.

## 2. Examples of the most popular machine learning algorithms in Python and R 

We'll be working on `iris` dataset, which is easily available in Python (`from sklearn import datasets; datasets.load_iris()`) and R (`data(iris)`).

We will use a few of the most popular machine learning tools: 

* R base, 
  
* R caret, 
  
    * [a short introduction to caret](https://cran.r-project.org/web/packages/caret/vignettes/caret.html)

    * [The *caret* package](http://topepo.github.io/caret/index.html)

* Python scikit-learn,
  
* Python API to [tensorflow](http://tomis9.com/tensorflow). `tensorflow` was used in very few cases, because it is designed mainly for neural networks, and we would have to implement the algorithms from scratch.

Let's prepare data for our algorithms. You can read more about data preparation in [this blog post](http://tomis9.com/useful_processing).

### data preparation 

*Python*
```{python, engine.path = '/usr/bin/python3'}
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

boston = datasets.load_boston()
# I will divide boston dataset to train and test later on
```

*tensorflow*

You can use functions from [tensorflow_datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3) module, but... does anybody use them?

*base R*
```{r, message = FALSE}
# unfortunately base R does not provide a function for train/test split
train_test_split <- function(test_size = 0.33, dataset) {
    smp_size <- floor(test_size * nrow(dataset))
    test_ind <- sample(seq_len(nrow(dataset)), size = smp_size)

    test <- dataset[test_ind, ]
    train <- dataset[-test_ind, ]
    return(list(train = train, test = test))
}

library(gsubfn)
list[train, test] <- train_test_split(0.5, iris)
```

*caret R*
```{r}
# docs: ..., the random sampling is done within the
# levels of ‘y’ when ‘y’ is a factor in an attempt to balance the class
# distributions within the splits.
# I provide package's name before function's name for clarity
```
```{r, message = FALSE}
trainIndex <- caret::createDataPartition(iris$Species, p=0.7, list = FALSE, 
                                         times = 1)
train <- iris[trainIndex,]
test <- iris[-trainIndex,]
```

### SVM 

The best description of SVM I found is in *Data mining and analysis - Zaki, Meira*. In general, I highly recommend this book. 

*Python*
```{python, engine.path = '/usr/bin/python3', message = FALSE, warning = FALSE}
from sklearn.svm import SVC  # support vector classification

svc = SVC()
svc.fit(X_train, y_train)

print(accuracy_score(svc.predict(X_test), y_test))
```

TODO: [plotting svm in scikit](https://scikit-learn.org/0.18/auto_examples/svm/plot_iris.html)

*"base" R*
```{r}
svc <- e1071::svm(Species ~ ., train)

pred <- as.character(predict(svc, test[, 1:4]))
mean(pred == test["Species"])
```

*caret R*
```{r, message = FALSE}
svm_linear <- caret::train(Species ~ ., data = train, method = "svmLinear")
mean(predict(svm_linear, test) == test$Species)
```

### decision trees 

*Python*
```{python, engine.path = '/usr/bin/python3'}
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

print(accuracy_score(y_test, dtc.predict(X_test)))
```

TODO: [an article on drawing decision trees in Python](https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176)

*"base" R*
```{r}
dtc <- rpart::rpart(Species ~ ., train)
print(dtc)
rpart.plot::rpart.plot(dtc)
pred <- predict(dtc, test[,1:4], type = "class")
mean(pred == test[["Species"]])
```

*caret R*
```{r}
c_dtc <- caret::train(Species ~ ., train, method = "rpart")
print(c_dtc)
rpart.plot::rpart.plot(c_dtc$finalModel)
```

I described working with decision trees in R in more detail in [another blog post](http://tomis9.com/decision_trees/).


### random forests 

*Python*
```{python, engine.path = '/usr/bin/python3', message = FALSE, warning = FALSE}
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

print(accuracy_score(rfc.predict(X_test), y_test))
```

*"base" R*
```{r}
rf <- randomForest::randomForest(Species ~ ., data = train)
mean(predict(rf, test[, 1:4]) == test[["Species"]])
```

*caret R*
```{r}
c_rf <- caret::train(Species ~ ., train, method = "rf")
print(c_rf)
print(c_dtc$finalModel)
```


### knn 

*Python*
```{python, engine.path = '/usr/bin/python3'}
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X, y)
print(accuracy_score(y, knn.predict(X)))
```

*R*
```{r}
kn <- class::knn(train[,1:4], test[,1:4], cl = train[,5], k = 3) 
mean(kn == test[,5])
```

TODO: caret r knn


### kmeans 

K-means can be nicely plotted in two dimensions with help of [PCA](http://tomis9.com/dimensionality/#pca).

*R*
```{r}
pca <- prcomp(iris[,1:4], center = TRUE, scale. = TRUE)
# devtools::install_github("vqv/ggbiplot")
ggbiplot::ggbiplot(pca, obs.scale = 1, var.scale = 1, groups = iris$Species, 
                   ellipse = TRUE, circle = TRUE)

iris_pca <- scale(iris[,1:4]) %*% pca$rotation 
iris_pca <- as.data.frame(iris_pca)
iris_pca <- cbind(iris_pca, Species = iris$Species)

ggplot2::ggplot(iris_pca, aes(x = PC1, y = PC2, color = Species)) +
  geom_point()

plot_kmeans <- function(km, iris_pca) {
  # we choose only first two components, so they could be plotted
  plot(iris_pca[,1:2], col = km$cluster, pch = as.integer(iris_pca$Species))
  points(km$centers, col = 1:2, pch = 8, cex = 2)
}
par(mfrow=c(1, 3))
# we use 3 centers, because we already know that there are 3 species
sapply(list(kmeans(iris_pca[,1], centers = 3),
            kmeans(iris_pca[,1:2], centers = 3),
            kmeans(iris_pca[,1:4], centers = 3)),
       plot_kmeans, iris_pca = iris_pca)
```

interesting article - [kmeans with dplyr and broom](https://cran.r-project.org/web/packages/broom/vignettes/kmeans.html)

TODO: caret r - kmeans

TODO: python - kmeans


### linear regression 

*Python*
```{python, engine.path = '/usr/bin/python3'}
from sklearn.linear_model import LinearRegression
from sklearn import datasets

X, y = boston.data, boston.target

lr = LinearRegression()
lr.fit(X, y)
print(lr.intercept_)
print(lr.coef_)
# TODO calculate this with iris dataset
```

*tensorflow*

- data preparation

```{python, engine.path = '/usr/bin/python3'}
import tensorflow as tf
from sklearn.datasets import load_iris
import numpy as np

def get_data(tensorflow=True):
    iris = load_iris()
    data = iris.data
    y = data[:, 0].reshape(150, 1)
    x0 = np.ones(150).reshape(150, 1)
    X = np.concatenate((x0, data[:, 1:]), axis=1)
    if tensorflow:
        y = tf.constant(y, name='y')
        X = tf.constant(X, name='X')  # constant is a tensor
    return X, y
```

- using normal equations

```{python, engine.path = '/usr/bin/python3'}
def construct_beta_graph(X, y):
    cov = tf.matmul(tf.transpose(X), X, name='cov')
    inv_cov = tf.linalg.inv(cov, name='inv_cov')
    xy = tf.matmul(tf.transpose(X), y, name='xy')
    beta = tf.matmul(inv_cov, xy, name='beta')
    return beta


X, y = get_data()
beta = construct_beta_graph(X, y)
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(beta.eval())
    print(mse.eval())
```

- using gradient descent and mini-batches
```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01
n_iter = 1000

X_train, y_train = get_data(tensorflow=False)

X = tf.placeholder("float64", shape=(None, 4))  # placeholder -
y = tf.placeholder("float64", shape=(None, 1))

start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(mse)

batch_indexes = np.arange(150).reshape(5,30)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        for batch_index in batch_indexes:
            _training.run(feed_dict={X: X_train[batch_index],
                                     y: y_train[batch_index]})
        if not i % 100:
            print(mse.eval(feed_dict={X: X_train, y: y_train}))
    print(mse.eval(feed_dict={X: X_train, y: y_train}), "- final score")
    print(beta.eval())
```

*base R*

```{r}
model <- lm(Sepal.Length ~ ., train)
summary(model)
```

`lm()` function automatically converts factor variables to one-hot encoded features.

*R caret*
```{r}
library(caret)

m <- train(Sepal.Length ~ ., data = train, method = "lm")
summary(m)  # exactly the same as lm()
```


### logistic regression 

In these examples I will present classification of a dummy variable.

*Python*
```{python, engine.path = '/usr/bin/python3'}
from sklearn.linear_model import LogisticRegression

cond = iris.target != 2
X = iris.data[cond]
y = iris.target[cond]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

lr = LogisticRegression()

lr.fit(X_train, y_train)
accuracy_score(lr.predict(X_test), y_test)
```

*base R*
```{r}
species <- c("setosa", "versicolor")
d <- iris[iris$Species %in% species,]
d$Species <- factor(d$Species, levels = species)
library(gsubfn)
list[train, test] <- train_test_split(0.5, d)

m <- glm(Species ~ Sepal.Length, train, family = binomial)
# predictions - if prediction is bigger than 0.5, we assume it's a one, 
# or success
y_hat_test <- predict(m, test[,1:4], type = "response") > 0.5

# glm's doc:
# For ‘binomial’ and ‘quasibinomial’ families the response can also
# be specified as a ‘factor’ (when the first level denotes failure
# and all others success) or as a two-column matrix with the columns
# giving the numbers of successes and failures.
# in our case - species[1] ("setosa") is a failure (0) and species[2] 
# ("versicolor") is 1 (success)
# successes:
y_test <- test$Species == species[2]

mean(y_test == y_hat_test)
```

*R caret*
```{r}
library(caret)
m2 <- train(Species ~ Sepal.Length, data = train, method = "glm", family = binomial)
mean(predict(m2, test) == test$Species)
```


### xgboost 

*Python*
```{python, engine.path = '/usr/bin/python3'}
from xgboost import XGBClassifier

cond = iris.target != 2
X = iris.data[cond]
y = iris.target[cond]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
accuracy_score(xgb.predict(X_test), y_test)
```

*"base" R*
```{r}
species <- c("setosa", "versicolor")
d <- iris[iris$Species %in% species,]
d$Species <- factor(d$Species, levels = species)
library(gsubfn)
list[train, test] <- train_test_split(0.5, d)

library(xgboost)
m <- xgboost(
  data = as.matrix(train[,1:4]), 
  label = as.integer(train$Species) - 1,
  objective = "binary:logistic",
  nrounds = 2)

mean(predict(m, as.matrix(test[,1:4])) > 0.5) == (as.integer(test$Species) - 1)
```

TODO: R: xgb.save(), xgb.importance()

*caret R*

TODO: [tuning xgboost with caret](https://www.kaggle.com/pelkoja/visual-xgboost-tuning-with-caret)
