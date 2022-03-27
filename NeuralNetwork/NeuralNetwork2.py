import tensorflow as tf
#keras = API, allows writing less code
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#load in the data set
data = keras.datasets.fashion_mnist

#split data into training and testing data
#the labels are between 0 and 9
(train_images, train_labels), (test_images, test_labels) = data.load_data()

print(train_labels[6])
print(tf.__version__)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#scale the values to be between 0 and 1
train_images = train_images/255.0
test_images = test_images/255.0

print(train_images[7])
plt.imshow(train_images[7])
plt.show()#display the image
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()#display the image

"""
To verify that the data is in the correct format and that you're ready 
to build and train the network, let's display the first 25 images from 
the training set and display the class name below each image.
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#flatten the data, each image is an array of arrays (matrix of 28x28 = 784 elements)
#by flattening the data instead of having a metrix we have a 1d-array of 784 elements
#which will be fed to the model

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"), #relu = rectify linear unit
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

#train the model
model.fit(train_images, train_labels, epochs=5)

#predict on one image, let's say the 7th
#prediction = model.predict([test_images[7]])

#predict on all images
prediction = model.predict(test_images)

#show input and the predicted value
#loop through a few images, show what they are and the predictions
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

#argmax - gives the largest value and returns the index of that
#to see the actual name and not the index, we pass the class_names of that index
print(class_names[np.argmax(prediction[0])])
