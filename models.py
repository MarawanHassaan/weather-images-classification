########################################## Imports #####################################################################
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from keras_preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.metrics import categorical_accuracy


TRAIN_SET_PATH = 'train'
TEST_SET_PATH = 'test'
batch_size = 32
######################################### Readers ######################################################################
def read_train_set():
    train_datagen = ImageDataGenerator( #Generate batches of tensor image data with real-time data augmentation.
        rescale=1./255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=10,
        vertical_flip=False,
        horizontal_flip=True
        )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_SET_PATH, #directory: string, path to the directory to read images from.
        target_size=(224, 224), #target_size: tuple of integers `(height, width)`, default: `(256, 256)` The dimensions to which all images found will be resized.
        batch_size=batch_size, #batch_size: size of the batches of data (default: 32)
        color_mode='rgb', #color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb". Whether the images will be converted to have 1 or 3 color channels.
        class_mode='categorical',
        shuffle=True
        ) #categorical"`: 2D numpy array of one-hot encoded labels.Supports multi-label output.
    return train_generator
def read_test_set():

    test_datagen = ImageDataGenerator(
        rescale=1./255
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_SET_PATH,
        target_size=(224, 224),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=False)
    return test_generator
def read_the_train_val_data():
    train_datagen = ImageDataGenerator(  # Generate batches of tensor image data with real-time data augmentation.
        rescale=1. / 255,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        rotation_range=10,
        vertical_flip=False,
        horizontal_flip=True,
        validation_split=0.2  #####################NEW#################################
    )
    train_generator = train_datagen.flow_from_directory(
        TRAIN_SET_PATH,  # directory: string, path to the directory to read images from.
        target_size=(224, 224),
        # target_size: tuple of integers `(height, width)`, default: `(256, 256)` The dimensions to which all images found will be resized.
        batch_size=batch_size,  # batch_size: size of the batches of data (default: 32)
        color_mode='rgb',
        # color_mode: one of "grayscale", "rgb", "rgba". Default: "rgb". Whether the images will be converted to have 1 or 3 color channels.
        class_mode='categorical',
        shuffle=True,
        subset='training')  # categorical"`: 2D numpy array of one-hot encoded labels.Supports multi-label output. #####################NEW#################################
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_SET_PATH,  # same directory as training data
        target_size=(224, 224),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical',
        shuffle=True,
        subset='validation')
    return train_generator,validation_generator
########################################################################################################################
######################################## CNN Models ####################################################################
'''
Models
'''
def AlexNet(input_shape, num_classes, regl2=0.0001, lr=0.0001):
    model = tf.keras.Sequential()

    # C1 Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=96, input_shape=input_shape, kernel_size=(11, 11), strides=(2, 4), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    # Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation before passing it to the next layer
    model.add(tf.keras.layers.BatchNormalization())


    # C2 Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    # Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # C3 Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # C4 Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # C5 Convolutional Layer
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(tf.keras.layers.Activation('relu'))
    # Pooling
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # Flatten
    model.add(tf.keras.layers.Flatten())

    flatten_shape = (input_shape[0] * input_shape[1] * input_shape[2],)

    # D1 Dense Layer
    model.add(tf.keras.layers.Dense(4096, input_shape=flatten_shape, kernel_regularizer=tf.keras.regularizers.l2(regl2)))
    model.add(tf.keras.layers.Activation('relu'))
    # Dropout
    model.add(tf.keras.layers.Dropout(0.4))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # D2 Dense Layer
    model.add(tf.keras.layers.Dense(4096, kernel_regularizer=tf.keras.regularizers.l2(regl2)))
    model.add(tf.keras.layers.Activation('relu'))
    # Dropout
    model.add(tf.keras.layers.Dropout(0.4))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # D3 Dense Layer
    model.add(tf.keras.layers.Dense(1000, kernel_regularizer=tf.keras.regularizers.l2(regl2)))
    model.add(tf.keras.layers.Activation('relu'))
    # Dropout
    model.add(tf.keras.layers.Dropout(0.4))
    # Batch Normalisation
    model.add(tf.keras.layers.BatchNormalization())

    # Output Layer
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))

    # Compile

    adam = tf.keras.optimizers.Adam(lr=lr)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model
def cnn(input_shape, num_classes):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.1))

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
def LeNet(classes):
    # source: https://github.com/f00-/mnist-lenet-keras/blob/master/lenet.py
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(20, (5, 5), padding='same', input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(50, (5, 5), padding='same'))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.Activation('relu'))

    model.add(tf.keras.layers.Dense(classes))
    model.add(tf.keras.layers.Activation('softmax'))

    return model
def vgg16(num_classes):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(input_shape=(224, 224, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096))
    model.add(tf.keras.layers.Activation('relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(num_classes))
    model.add(tf.keras.layers.Activation('softmax'))
    #sgd = tf.keras.optimizers.SGD(lr=0.0001, decay=0, nesterov=True)
    opt=tf.keras.optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model
