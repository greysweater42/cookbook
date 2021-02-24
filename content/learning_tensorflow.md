---
title: "learning tensorflow"
date: 2019-08-05T18:45:18+02:00
draft: false
categories: ["Python", "Machine learning"]
---

The best way to gain intuiton to any new thing you learn is to start from a very beginning and play with it (*let's see what happens if I do this*). That's the power of reinforcement learning ;)

These packages will be useful in the nearest future:

```{python, engine.path = '/usr/bin/python3'}
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

import tensorflow as tf
```

A trivial example of tensorflow:

```{python, engine.path = '/usr/bin/python3'}
import tensorflow as tf

a = tf.Variable(10, name='a')  # a variable
b = tf.Variable(12, name='b')  # another variable

s = a + b  # a tensor

sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
print(sess.run(s))
```

As you can see, one does not simply add two numbers in tensorflow.

* lesson #1: you cannot initialize a tensor

```{python, engine.path = '/usr/bin/python3'}
sess = tf.Session()
sess.run(a.initializer)
sess.run(b.initializer)
# print(s.eval())  # does not work - you eval() is not connected to the session
# anyhow
print(sess.run(s))
```

* lesson #2: eval does not recognize session by itself

```{python, engine.path = '/usr/bin/python3'}
with tf.Session() as sess:
    sess.run(a.initializer)
    sess.run(b.initializer)
    print(s.eval())  # does work - eval recognizes a default session
```

* lesson #2: eval works in `with` clause

* lesson #3: so far there are 2 ways to initialize a variable in a session

```{python, engine.path = '/usr/bin/python3'}
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    print(s.eval())
```

* lesson #4: the third and most compact way to initialize variables - all the variables in one statement

```{python, engine.path = '/usr/bin/python3'}
s1 = tf.add(a, b)  # a tensor, not variable
with tf.Session() as sess:
    init.run()
    print(s1.eval())
```

* lesson #5: tensorflow has it's own mathematical functions

```{python, engine.path = '/usr/bin/python3'}
c = tf.Variable(np.array([[1, 2], [3, 4]]), name='c')
d = tf.Variable(np.array([[5, 6], [7, 8]]), name='d')

m = tf.matmul(c, d)  # a tensor again

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(m.eval())
```

* lesson #6: tensorflow can interpret numpy arrays as matrices
* lesson #7: you can multiply matrices!

```{python, engine.path = '/usr/bin/python3'}
iris = load_iris()
data = iris.data
y = tf.Variable(data[:, 0].reshape(150, 1), name='y')
x0 = np.ones(150).reshape(150, 1)
x0_X = np.concatenate((x0, data[:, 1:]), axis=1)
X = tf.Variable(x0_X, name='X')

cov = tf.matmul(tf.transpose(X), X, name='cov')
inv_cov = tf.matrix_inverse(cov, name='inv_cov')
xy = tf.matmul(tf.transpose(X), y, name='xy')
beta = tf.matmul(inv_cov, xy)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(beta.eval())


lr = LinearRegression()
lr.fit(data[:, 1:], data[:, 0])
print(np.concatenate((np.array([lr.intercept_]), lr.coef_)))
```

* lesson #8: when creating a tensorflow vector, ou have to provide information if it's horizontal or mathematical, just like in mathematics lesson #9: tensorflow gives the same results as sklearn (linear regression)

the code above looks quite like a mess, let's clear it up

```{python, engine.path = '/usr/bin/python3'}
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


def construct_beta_graph(X, y):
    cov = tf.matmul(tf.transpose(X), X, name='cov')
    inv_cov = tf.matrix_inverse(cov, name='inv_cov')
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

* lesson #10: you can easily divide your code into modules to make it easier to read lesson #11: when dealing with input data, you can use tf.constant instead of tf.Variable, as the data never changes; constant is a tensor

```{python, engine.path = '/usr/bin/python3'}
X, y = get_data()
learning_rate = 0.0001

beta = tf.Variable(np.random.rand(4).reshape(4, 1))
gradient = tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)
new_beta = beta - learning_rate * gradient

mse_old = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))
mse_new = tf.reduce_mean(tf.square(y - tf.matmul(X, new_beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    print(beta.eval())
    print(new_beta.eval())
    print(mse_old.eval())
    print(mse_new.eval())
```

* lesson #12: you can calculate the gradient of mse pretty simply on a piece of paper

```{python, engine.path = '/usr/bin/python3'}
X, y = get_data()
learning_rate = 0.01

beta = tf.Variable(np.random.rand(4).reshape(4, 1))
gradient = 2 / 150 * tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)

_training = tf.assign(beta, beta - learning_rate * gradient)

mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(100):
        _training.eval()
        print(mse.eval())
    print(beta.eval())
```

* lesson #13: tf.assign - assign one value to another, training tensor is only technical, so we could point that the assignment should be made in every iteration

let's clear the code a little bit


```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01
n_iter = 1000

X, y = get_data()
beta = tf.Variable(np.random.rand(4).reshape(4, 1))
gradient = 2 / 150 * tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)

_training = tf.assign(beta, beta - learning_rate * gradient)

mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        _training.eval()
        if not i % 100:
            print(mse.eval())
    print(mse.eval())
    print(beta.eval())
```

we can make this even better if we use different starting values in each run

```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01
n_iter = 10000

X, y = get_data()
start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

# gradient = 2 / 150 * tf.matmul(tf.transpose(X), tf.matmul(X, beta) - y)
gradient = tf.gradients(mse, [beta])[0]

_training = tf.assign(beta, beta - learning_rate * gradient)  # a tensor

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        _training.eval()
        if not i % 1000:
            print(mse.eval())
    print(mse.eval())
    print(beta.eval())
```

* lesson #14: it is better to use tensorflow starting values, as they change in every run
* lesson #15: you can calculate gradient manually by yourself, but you can use numerical algorithms implemented in tf.gradients

```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01
n_iter = 10000

X, y = get_data()
start_values = tf.random_uniform([4, 1], -1, 1, dtype="float64")
beta = tf.Variable(start_values, name='beta')
mse = tf.reduce_mean(tf.square(y - tf.matmul(X, beta)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(mse)  # an operation - a new class of objects

init = tf.global_variables_initializer()
with tf.Session() as sess:
    init.run()
    for i in range(n_iter):
        _training.run()  # operations are being run, not evaluated
        if not i % 1000:
            print(mse.eval())
    print(mse.eval())
    print(beta.eval())
```

* lesson #16: an optimizer knows, that it can change variables, not constants
* lesson #17: operations (like optimizer) are run, not evaluated


you should always use get_variable() insetad of Variable (interesting) [link to stackoverflow discussion](https://stackoverflow.com/questions/37098546/difference-between-variable-and-get-variable-in-tensorflow)

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

batch_indexes = np.arange(150).reshape(5, 30)

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

* lesson #18: in mini-batch processing it is comfortable to use placeholders

```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01

iris = load_iris()
X_np, y_np = iris.data, iris.target

ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))

x = tf.placeholder(tf.float64, shape=(4, None))
y = tf.placeholder(tf.float64, shape=(3, None))

W = tf.Variable(tf.random_uniform([3, 4], -1, 1, dtype="float64"))
b = tf.Variable(tf.random_uniform([3, 1], -1, 1, dtype="float64"))

mult = tf.matmul(W, x) + b  # broadcasting just like in numpy
y_hat = tf.nn.softmax(mult, axis=0)

error = tf.reduce_mean(tf.square(y - y_hat))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(error)

init = tf.global_variables_initializer()

batches = np.arange(150).reshape(5, 30)

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch].transpose(),
                                     y: y_all[batch].transpose()})
        if not i % 100:
            print(error.eval(feed_dict={x: X_np.transpose(),
                                        y: y_all.transpose()}))
    preds = y_hat \
        .eval(feed_dict={x: X_np.transpose()}) \
        .transpose()

def calculate_accuracy(preds):
    preds_max = np.amax(preds, axis=1)

    max_indexes = []
    for pred, pred_max in zip(preds, preds_max):
        prediction = np.where(pred == pred_max)[0][0]
        max_indexes.append(prediction)

    preds_cat = np.array(max_indexes)
    return(accuracy_score(y_np, preds_cat))

calculate_accuracy(preds)  # maybe overfitting?
```

* lesson #19: tf.reshape is NOT the same as tf.transpose
* lesson #20: tf.nn.softmax works on rows, not columns. Oh, that's nice. You can provide "axis" parameter in this function
* lesson #21: in tensorflow you will find broadcasting, just like in numpy

* name_scope

```{python, engine.path = '/usr/bin/python3'}
with tf.name_scope("constants"):
    a = tf.constant(10, name='a')
    b = tf.constant(12, name='b')
```

* and variable scope

```{python, engine.path = '/usr/bin/python3'}
with tf.variable_scope("variables"):
    c = tf.constant(20, name='c')
    d = tf.constant(22, name='d')
```

```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01

iris = load_iris()
X_np, y_np = iris.data, iris.target

ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))

x = tf.placeholder(tf.float64, shape=(None, 4), name='x')
y = tf.placeholder(tf.float64, shape=(None, 3), name='y')

W0 = tf.Variable(tf.random_uniform([4, 3], -1, 1, dtype=tf.float64), name='W0')
b0 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64), name='b0')  # will broadcast
h = tf.nn.softmax(tf.matmul(x, W0) + b0)

W1 = tf.Variable(tf.random_uniform([3, 3], -1, 1, dtype=tf.float64), name='W1')
b1 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64), name='b1')
y_hat = tf.nn.softmax(tf.matmul(h, W1) + b1)

error = tf.reduce_mean(tf.square(y - y_hat))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
_training = optimizer.minimize(error)

init = tf.global_variables_initializer()

batches = np.arange(150).reshape(5, 30)

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch], y: y_all[batch]})
        if not i % 1000:
            print(error.eval(feed_dict={x: X_np, y: y_all}))
    preds = y_hat.eval(feed_dict={x: X_np})

print(calculate_accuracy(preds))  # maybe overfitting?
```

* lesson #21: deeper neaural networks converge much more slowly

```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01

iris = load_iris()
X_np, y_np = iris.data, iris.target

ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))

x = tf.placeholder(tf.float64, shape=(None, 4), name='x')
y = tf.placeholder(tf.float64, shape=(None, 3), name='y')

with tf.variable_scope('layer1'):
    W0 = tf.Variable(tf.random_uniform([4, 3], -1, 1, dtype=tf.float64))
    b0 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64))
    h = tf.nn.softmax(tf.matmul(x, W0) + b0)

with tf.variable_scope('layer2'):
    W1 = tf.Variable(tf.random_uniform([3, 3], -1, 1, dtype=tf.float64))
    b1 = tf.Variable(tf.random_uniform([1, 3], -1, 1, dtype=tf.float64))
    y_hat = tf.nn.softmax(tf.matmul(h, W1) + b1)

with tf.variable_scope('training'):
    error = tf.reduce_mean(tf.square(y - y_hat))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    _training = optimizer.minimize(error)

init = tf.global_variables_initializer()

batches = np.arange(150).reshape(5, 30)

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch], y: y_all[batch]})
        if not i % 1000:
            print(error.eval(feed_dict={x: X_np, y: y_all}))
    preds = y_hat.eval(feed_dict={x: X_np})

print(calculate_accuracy(preds))  # maybe overfitting?
```

```{python, engine.path = '/usr/bin/python3'}
learning_rate = 0.01

iris = load_iris()
X_np, y_np = iris.data, iris.target

ohe = OneHotEncoder(sparse=False)
y_all = ohe.fit_transform(y_np.reshape(len(y_np), 1))

x = tf.placeholder(tf.float64, shape=(None, 4), name='x')
y = tf.placeholder(tf.float64, shape=(None, 3), name='y')

def neural_layer(scope_name, x, input_size, output_size, func):
    with tf.variable_scope(scope_name):
        W_shape = [input_size, output_size]
        b_shape = [1, output_size]
        W = tf.Variable(tf.random_uniform(W_shape, -1, 1, dtype=tf.float64))
        b = tf.Variable(tf.random_uniform(b_shape, -1, 1, dtype=tf.float64))
        z = func(tf.matmul(x, W) + b)
    return z

h = neural_layer('layer1', x, 4, 3, tf.nn.relu)
y_hat = neural_layer('layer1', h, 3, 3, tf.nn.softmax)

with tf.variable_scope('training'):
    error = tf.reduce_mean(tf.square(y - y_hat))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    _training = optimizer.minimize(error)

init = tf.global_variables_initializer()

batches = np.arange(150).reshape(5, 30)

with tf.Session() as sess:
    sess.run(init)
    for i in range(10000):
        for batch in batches:
            _training.run(feed_dict={x: X_np[batch], y: y_all[batch]})
        if not i % 1000:
            print(error.eval(feed_dict={x: X_np, y: y_all}))
    preds = y_hat.eval(feed_dict={x: X_np})

print(calculate_accuracy(preds))  # maybe overfitting?
```
