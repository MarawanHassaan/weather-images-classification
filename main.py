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
########################################################################################################################

models_dir ='models'
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
########################################################################################################################
#train_generator = read_train_set()
#test_generator = read_test_set()
train_generator, validation_generator= read_the_train_val_data()

num_samples = train_generator.n
num_classes = train_generator.num_classes
input_shape = train_generator.image_shape
classnames = [k for k,v in train_generator.class_indices.items()]
print("Image input %s" %str(input_shape))
print("Classes: %r" %classnames)
print('Loaded %d training samples from %d classes.' %(num_samples,num_classes))
print('Loaded %d test samples from %d classes.' %(validation_generator.n,validation_generator.num_classes))



############################################# Model Creation ###########################################################
'''
AlexNet
'''
#model = AlexNet(input_shape, num_classes)
#print(model.summary())
'''
Tensorflow VGG16
'''
#model = tf.keras.applications.VGG16(weights=None, classes=len(train_generator.class_indices), input_shape=(224, 224, 3))
#print(model.summary())
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[categorical_accuracy])
'''
VGG16
'''
#model= vgg16(num_classes=4)
#print(model.summary())
'''
CNN
'''
#model=cnn(input_shape=(224,224,3),num_classes=4)
#print(model.summary())
#model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=[categorical_accuracy])

'''
LeNET
'''
model= LeNet(classes=4)
print(model.summary())
#opt=tf.keras.optimizers.SGD(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
################################################ Model Train ###########################################################

t0 = time.time()
steps_per_epoch=train_generator.n//train_generator.batch_size
val_steps=validation_generator.n//validation_generator.batch_size
model.fit_generator(train_generator, epochs=50, verbose=1,
                  steps_per_epoch=steps_per_epoch,\
                   validation_data=validation_generator,\
                    validation_steps=val_steps)

t1 = time.time()
print("Elapsed Training Time: {}".format(t1-t0))
############################################## Model Storing ###########################################################

savemodel(model, 'D1312LeNetTrialepoch50')


############################################## Model Loading ###########################################################

#model = loadmodel('vgg16')
#print(model.summary())


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