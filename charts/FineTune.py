import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

#############################################################
#############################################################
# Functions

# Function for plotting images in batch
def PlotImg(images_arr):
    fig, axes = plt.subplots(1, 4, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Function for creating & plotting confusion metrics
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#############################################################
#############################################################
# Batch Creation

# Defining paths of the train, validation, and test image sources
Train_Path = 'Train'
Val_Path = 'Val'
Test_Path = 'Test'

# Creating the Batches of Data for Train, Valid, and Test (that will pass through seq. model for prediction) also,
# ImageDataGenerator function label the data according shown classes [Dot, Hor Bar, etc.]
Train_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Train_Path, target_size=(224, 224),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'], batch_size=4)
Val_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Val_Path, target_size=(224, 224),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'], batch_size=4)
Test_Batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=Test_Path, target_size=(224, 224),
                         classes=['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar'],
                         batch_size=4, shuffle=False)


#############################################################
#############################################################
# Importing and Updating Pretrained Model (VGG16)

# Imported VGG16
vgg16_model = tf.keras.applications.vgg16.VGG16()

# Before Update
print('VGG16')
vgg16_model.summary()

# Remove One last layer of VGG16 and add our layer of 5 units, and freeze all weights instead last 5 neurons
print('Copy - VGG16')
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)
model.summary()

for layer in model.layers:
    layer.trainable = False

print('After Add')
model.add(Dense(units=5, activation='softmax'))
# After Update
model.summary()


#############################################################
#############################################################
# Train Model through Train and Validation data

# Compile model with low learning rate as proven optimal and cat.-crossentropy as we have multiple prediction option
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(x=Train_Batches, steps_per_epoch=len(Train_Batches), validation_data=Val_Batches,
                    validation_steps=len(Val_Batches), epochs=5, verbose=2)

#############################################################
#############################################################
# Prediction using test data

# creates the batch from test data
test_imgs, test_labels = next(Test_Batches)

# Do prediction for that batch
predictions = model.predict(x=Test_Batches, steps=len(Test_Batches), verbose=0)
np.round(predictions)

# Setting up for creating confusion metrics
cm = confusion_matrix(y_true=Test_Batches.classes, y_pred=np.argmax(predictions, axis=-1))

Test_Batches.class_indices

cm_plot_labels = ['Dot Line', 'Horizontal Bar', 'Line', 'Pie', 'Vertical Bar']


#############################################################
#############################################################
# Plotting Data

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

fig = plt.figure()
plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

#####################################################
#####################################################
# Save Model
model.save('UpdatedVGG16.h5')

plt.show()