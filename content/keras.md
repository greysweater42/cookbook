---
title: "keras"
date: 2021-04-08T18:19:38+01:00
draft: false
categories: ["Python", "Machine learning"]
tags: []
---


# 1. What is keras and why would you still use pytorch?

- Just kidding with [pytorch](https://greysweater42.github.io/pytorch) ;) but still you will have to choose between these two (or even more) frameworks at the very early stage of the analysis.

- Keras is a high-level framework for working with neural networks, written in Python and capable of running on top of either TensorFlow, Microsoft Cognitive Toolkit (CNTK) or Theano.

# 2. Example usage

## An absoolutely basic example

Yes, it's going to be MNIST ;) 

Based on [Deep learning with Python](https://www.manning.com/books/deep-learning-with-python), chapter 2.

You can use `Sequential API` or `Model API` (there is also Functional API, but I will not cover it as it is almost exactly the same Sequential API), they both work exactly the same, but as you can see, `Sequential API`'s syntax is much shorter and `Model API`'s syntax looks almost exactly the same as [pytorch](https://greysweater42.github.io/pytorch/#neural-net).

```
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape((60000, 28 * 28)).astype('float32') / 255
X_test = X_test.reshape((10000, 28 * 28)).astype('float32') / 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# you can use Sequential API
import keras
network = keras.models.Sequential()
network.add(keras.layers.Dense(512, activation='relu'))
network.add(keras.layers.Dense(10, activation='softmax'))

# or Model API - up to you!
from tensorflow import nn, keras

class Network(keras.Model):

  def __init__(self):
    super(Network, self).__init__()
    self.d1 = keras.layers.Dense(512, activation=nn.relu)
    self.d2 = keras.layers.Dense(10, activation=nn.softmax)

  def call(self, inputs):
    x = self.d1(inputs)
    return self.d2(x)

network = Network()
####
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
network.fit(X_train, y_train, epochs=5, batch_size=128)

test_loss, test_acc = network.evaluate(X_test, y_test)
print(test_acc)
y_hat = network.predict(X_test)  # and this is how you make predictions
```

## GPU support

The easiest way to run keras on GPU is to pull a tensorflow-gpu image with:

```
docker pull tensorflow/tensorflow:latest-gpu
```

start it up with:

```
docker run --gpus all -it --rm tensorflow/tensorflow:latest-gpu bash
```

and run your code in it. But first you should configure GPU on the machine you're working at. Just follow [this procedure](https://www.tensorflow.org/install/gpu) (basically you should get to the point, when you can run `nvidia-smi` from your terminal).

# 3. Interesting resources

- if you still prefer R over Python, take a look at this introductory article: [keras: deep learning in R](https://www.datacamp.com/community/tutorials/keras-r-deep-learning)

- a very good book not only about keras, but about deep learning in general: [Deep learning with Python](https://www.manning.com/books/deep-learning-with-python)

# 4. Subjects still to cover

- is there an easy way to read images into the network? TODO yes, with ImageDataGenerator

- saving and loading networks? TODO

- transfer learning? TODO

- data augmentation? TODO

- YOLO or object detection, in general TODO

