import cv2
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt


# paths
folder_path = 'test-set/'
label_path = 'labels.csv'

# labels array
labels = np.genfromtxt(label_path, delimiter=',')

# variables
count = 0
num_correct = 0

# load model
model = tf.keras.models.load_model('classifier-model.h5')

# get each file name in sorted directory
for filename in sorted(os.listdir(folder_path)):
    # read in image
    img = cv2.imread(os.path.join(folder_path, filename))
    # resize image to 28x28
    img = cv2.resize(img, (28, 28))
    # change to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # invert image, mnist encodes white w/ 0, and black w/ 255 (while normally its opposite)
    img = cv2.bitwise_not(img)
    # mean normalization
    img = (img / 255) - 0.5

    # display image
    plt.imshow(img, cmap='Greys')
    plt.show()

    # flatten the image
    img = img.reshape((-1, 784))

    print(filename)
    # correct label
    correct = int(labels[count])
    # predict on image
    prediction = model.predict(img)
    # predicted label
    output = np.argmax(prediction[0])

    print("actual value of image: " + str(correct))
    print("predicted value of image: " + str(output))

    if correct == output:
        print("model predicted correctly!")
        num_correct += 1
    else:
        print("model predicted incorrectly...")

    print()
    count += 1

# accuracy of model
accuracy = float(num_correct)/count

print("accuracy of model on hand-drawn set: " + str(accuracy))