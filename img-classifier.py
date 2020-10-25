import numpy as np
import os
import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# training sets
train_images = mnist.train_images()
train_labels = mnist.train_labels()

# test sets
test_images = mnist.test_images()
test_labels = mnist.test_labels()

# mean normalization
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5


# flattening our inputs into 1d vectors
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

# constructing our model
model = Sequential([
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax'),
])

# compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# load the model from disk if there exists previous weights
if os.path.isfile('image-classifier.h5'):
    model.load_weights('image-classifier.h5')


# 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# training the model
model.fit(
    train_images,
    to_categorical(train_labels),
    epochs=10,
    batch_size=32
)

# save model weights
model.save_weights('image-classifier.h5')

# evaluate the model
model.evaluate(
    test_images,
    to_categorical(test_labels)
)

# predict on the first 10 images
predictions = model.predict(test_images[:10])

# print actual first 10 digits from test set
print("actual digits:")
print(test_labels[:10])

# print model's predictions, to compare
print("predicted digits:")
print(np.argmax(predictions, axis=1))

# save model to disk
model.save('classifier-model.h5')


