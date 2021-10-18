import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import text_dataset_from_directory
from keras_prepare_data import DATA_PATH

VOCAB_SIZE = 10000

train = text_dataset_from_directory(
    DATA_PATH / "train", labels="inferred", batch_size=32
)
test = text_dataset_from_directory(DATA_PATH / "test", labels="inferred", batch_size=32)

encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
encoder.adapt(train.map(lambda text, label: text))
vocab = np.array(encoder.get_vocabulary())

model = tf.keras.Sequential(
    [
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=300,
            mask_zero=True,
        ),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1),
    ]
)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(1e-4),
    metrics=["accuracy"],
)
history = model.fit(train, epochs=3, validation_data=test, validation_steps=30)
print("accuracy: ",  history.history['val_accuracy'][-1])
# accuracy:  0.987500011920929

ls = []
y_hat = []
for texts, labels in test:
  y_hat.append(model.predict(texts).reshape(-1)) 
  ls.append(labels.numpy())

ls = np.concatenate(ls)
y_hat = np.concatenate(y_hat)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ls, y_hat > 0))
# [[ 347    5]
#  [  14 1185]]
