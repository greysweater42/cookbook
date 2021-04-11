---
title: "keras"
date: 2021-04-08T18:19:38+01:00
draft: false
categories: ["Python", "Machine learning"]
tags: []
---


# 1. What is keras and why would you still use pytorch?

- Just kidding with [pytorch](https://greysweater42.github.io/pytorch) ;) but still you will have to choose between these two (or even more) frameworks at the very early stage of the analysis.

- Keras is a high-level framework for working with neural networks, written in Python and capable of running on top of either TensorFlow, Microsoft Cognitive Toolkit (CNTK) or Theano

# 2. Example usage

## An absoolutely basic example

Yes, it's going to be MNIST ;) 

Based on [Deep learning with Python](https://www.manning.com/books/deep-learning-with-python), chapter 2.

```
from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000, 28 * 28)).astype('float32') / 255
X_test = X_test.reshape((10000, 28 * 28)).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from keras import models, layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


network.fit(X_train, y_train, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(X_test, y_test)
# y_hat = network.predict(X_test)  # and this is how you make predictions
```

## GPU support

TO be honest, I've enver used any other backend than tensorflow and I don't see any particular reason for doing otherwise. Keras does not provide any interface to force calculation on GPU, so we have to go under the hood, straight to tensorflow.

```
import tensorflow as tf
from tensorflow.python.keras import backend as K

# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
```

Solution provided by [stackoverflow](https://stackoverflow.com/a/66047607).

You can test it on [Google Colab](https://colab.research.google.com/), which provides modest, but free GPU in cloud to have fun with. Just remebmber to initiate GPU support in Edit > Notebook settings > HArdware accelerator > GPU.

# 3. Interesting resources

- if you still prefer R over Python, take a look at this introductory article: [keras: deep learning in R](https://www.datacamp.com/community/tutorials/keras-r-deep-learning)

- a very good book not only about keras, but about deep learning in general: [Deep learning with Python](https://www.manning.com/books/deep-learning-with-python)

# 4. Subjects still to cover

- is there an easy way to read images into the network? TODO yes, with ImageDataGenerator

- saving and loading networks? TODO

- transfer learning? TODO

- data augmentation? TODO

