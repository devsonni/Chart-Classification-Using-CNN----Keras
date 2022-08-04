import numpy as np
import tensorflow as tf
from tensorflow import keras
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
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'], batch_size=3)
Val_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Val_Path, target_size=(128, 128),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'], batch_size=3)
Test_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Test_Path,
                         target_size=(128, 128),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'],
                         batch_size=3, shuffle=False)

# Creates the batch and takes the labels
imgs, labels = next(Train_Batches)

# function for plotting images
def PlotImg(images_arr):
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Testing how all goes, all seems perfect getting correct correct labels with images
PlotImg(imgs)
print(labels)

