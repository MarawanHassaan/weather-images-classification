########################################## Imports #####################################################################
import numpy as np
import time
import os
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from keras_preprocessing.image import ImageDataGenerator
from tensorflow_core.python.keras.metrics import categorical_accuracy
from models import *

TRAIN_SET_PATH = 'train'
TEST_SET_PATH = 'test'
batch_size = 32
models_dir ='models_part2'
def savemodel(model,problem):
    filename = os.path.join(models_dir, '%s.h5' %problem)
    model.save(filename)
    print("\nModel saved successfully on file %s\n" %filename)
def loadmodel(problem):
    filename = os.path.join(models_dir, '%s.h5' %problem)
    try:
        model = tf.keras.models.load_model(filename)
        print("\nModel loaded successfully from file %s\n" %filename)
    except OSError:
        print("\nModel file %s not found!!!\n" %filename)
        model = None
    return model

###################################### Load the data ###################################################################
train_generator, validation_generator= read_the_train_val_data()
num_samples = train_generator.n
num_classes = train_generator.num_classes
input_shape = train_generator.image_shape
classnames = [k for k,v in train_generator.class_indices.items()]
print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)
print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(validation_generator.n,validation_generator.num_classes))


'''
############################################# Model Creation ###########################################################
def load_backbone_net(input_shape):
    # define input tensor
    input0 = tf.keras.Input(shape=input_shape)

    # load a pretrained model on imagenet without the final dense layer
    feature_extractor = tf.keras.applications.VGG16(include_top=False, weights='imagenet', input_tensor=input0)

    feature_extractor = feature_extractor.output
    feature_extractor = tf.keras.models.Model(inputs = input0, outputs = feature_extractor)
    optimizer = 'adam'  # alternative 'SGD'

    feature_extractor.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return feature_extractor
def transferNet(feature_extractor, num_classes, output_layer_name, trainable_layers):
    # get the original input layer tensor
    input_t = feature_extractor.get_layer(index=0).input

    # set the feture extractor layers as non-trainable
    for idx, layer in enumerate(feature_extractor.layers):
        if layer.name in trainable_layers:
            layer.trainable = True
        else:
            layer.trainable = False

    # get the output tensor from a layer of the feature extractor
    output_extractor = feature_extractor.get_layer(name=output_layer_name).output

    # output_extractor = MaxPooling2D(pool_size=(4,4))(output_extractor)

    # flat the output of a Conv layer
    flatten = tf.keras.layers.Flatten()(output_extractor)
    flatten_norm = tf.keras.layers.BatchNormalization()(flatten)

    # add a Dense layer
    dense = tf.keras.layers.Dropout(0.4)(flatten_norm)
    dense = tf.keras.layers.Dense(200, activation='relu')(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)

    # add a Dense layer
    dense = tf.keras.layers.Dropout(0.4)(dense)
    dense = tf.keras.layers.Dense(100, activation='relu')(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)

    # add the final output layer
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.Dense(num_classes, activation='softmax')(dense)

    model = tf.keras.models.Model(inputs=input_t, outputs=dense, name="transferNet")

    optimizer = 'adam'  # alternative 'SGD'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model
# choose the layer from which you can get the features (block5_pool the end, glob_pooling to get the pooled version of the output)


# load the pre-trained model
feature_extractor = load_backbone_net(input_shape)
feature_extractor.summary()
name_output_extractor = "block5_pool"
trainable_layers = ["block5_conv3"]
# build the transfer model
transfer_model = transferNet(feature_extractor, num_classes, name_output_extractor, trainable_layers)
transfer_model.summary()

################################################ Model Train ###########################################################
stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)
steps_per_epoch = train_generator.n//train_generator.batch_size
val_steps = validation_generator.n//validation_generator.batch_size
t0 = time.time()
transfer_model.fit_generator(train_generator, epochs=50, callbacks=[stopping],
                             verbose=1,steps_per_epoch=steps_per_epoch,
                             validation_data=validation_generator,validation_steps=val_steps)


t1 = time.time()
print("Elapsed Training Time: {}".format(t1-t0))
############################################## Model Storing ###########################################################
savemodel(transfer_model, 'D1412Trial2')
'''


############################################## Model Loading ###########################################################

model = loadmodel('D1412Trial2')
print(model.summary())


################################################ Evaluation ############################################################
test_datagen = ImageDataGenerator(
    rescale = 1. / 255)
test_generator = test_datagen.flow_from_directory(
    TEST_SET_PATH,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False)
val_steps=test_generator.n//test_generator.batch_size+1
loss, acc = model.evaluate_generator(test_generator, verbose=1, steps=val_steps)
print('Test loss: %f' %loss)
print('Test accuracy: %f' %acc)

##############################Precision/Recall/f1score & Confusion Matrix ##############################################
y_pred = model.predict_generator(test_generator, steps=val_steps, verbose=1)
y_pred = np.argmax(y_pred, axis=1)
y_test = test_generator.classes
print(classification_report(y_test, y_pred, labels=None, target_names=classnames, digits=3))

import sklearn.metrics
from sklearn.metrics import classification_report, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


########################################################################################################################
########################################################################################################################
########################################################################################################################


''''
NEW
'''

#BLIND_SET_PATH = 'blind'
#blind_datagen = ImageDataGenerator(
#    rescale = 1. / 255)
#test_generator = blind_datagen.flow_from_directory(
#    BLIND_SET_PATH,
#    target_size=(224, 224),
#    color_mode="rgb",
#    batch_size=batch_size,
#    class_mode="categorical",
#    shuffle=False)
#y_pred = model.predict_generator(blind_generator, steps=val_steps, verbose=1)
#y_pred = np.argmax(y_pred, axis=1)
################################################ CSV ###################################################################

#import csv
#with open('blind.csv', mode='w', newline='') as f:
#	out=csv.writer(f)
#	out.writerows(map(lambda x: [x], y_pred))
'''
End
'''
