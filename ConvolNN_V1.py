import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
import shutil
import random
import glob
import warnings

# Defining paths of the train, validation, and test image sources
Train_Path = 'charts/Train'
Val_Path = 'charts/Val'
Test_Path = 'charts/Test'

# Creating the Batches of Data for Train, Valid, and Test (that will pass through seq. model for prediction) also,
# ImageDataGenerator function label the data according shown classes [Dot, Hor Bar, etc.]
Train_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Train_Path, target_size=(128, 128),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'], batch_size=4)
Val_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Val_Path, target_size=(128, 128),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'], batch_size=4)
Test_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Test_Path, target_size=(128, 128),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'],
                         batch_size=4, shuffle=False)


# Creates the batch and takes the labels
# imgs, labels = next(Train_Batches)

# function for plotting images
def PlotImg(images_arr):
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# Testing how all goes, all seems perfect getting correct correct labels with images
# PlotImg(imgs)
# print(labels)

# Creating 2 layered CNN Model, where first layer is 2D con layer with 32 filters of 3*3 sized with 2*2 MaxPooling \
# layer with stride of 2, and second layer has 64 same sized filters and same pooling layer as before.
# and we're also adding padding, so, padding = same
model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=5, activation='softmax')
])

# Checkout
model.summary()
# Compile model with low learning rate as proven optimal and cat.-crossentropy as we have multiple prediction option
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=Train_Batches, steps_per_epoch=len(Train_Batches), validation_data=Val_Batches,
                    validation_steps=len(Val_Batches), epochs=10, verbose=2)

fig = plt.figure()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')

fig = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()
