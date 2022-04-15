from logging.config import valid_ident
from pickletools import optimize
from statistics import mode
import keras,os
from keras.optimizers import rmsprop_v2
from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.layers import  Dropout,Flatten,Dense,Activation,MaxPool2D,Conv2D
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.python.keras import backend as K
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib as plt
tf.keras.optimizers.SGD(
    learning_rate=0.0001, 
    momentum=0.0, 
    nesterov=False, 
    name="SGD"
)
tf.keras.optimizers.Adam(
    learning_rate=0.00001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=0.00000001,
    amsgrad=False,
    name='Adam',
)
tf.keras.optimizers.RMSprop(
    learning_rate=0.001,
    rho=0.9,
    momentum=0.0,
    epsilon=1e-07,
    centered=False,
    name="RMSprop"
)
tf.keras.optimizers.Adamax(
    learning_rate=0.001,
    beta_1=0.9, 
    beta_2=0.999, 
    epsilon=1e-07, 
    name="Adamax"
)
K.clear_session()
K._get_available_gpus()
Train_data = ImageDataGenerator()
Val_Data = ImageDataGenerator()
Image_train = Train_data.flow_from_directory(
    directory="./Image/train",
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)
Image_Val = Val_Data.flow_from_directory(
    directory="./Image/test",
    batch_size=12,
    target_size=(224,224),
    class_mode='categorical'
)
##
model = Sequential()
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
##
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=5, activation="softmax"))
##
model.summary()
model.compile(optimizer='Adamax', loss= categorical_crossentropy, metrics=['accuracy'])
checkpoint = ModelCheckpoint("CofferDM.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=30, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=200,generator=Image_train, validation_data= Image_Val, validation_steps=100,epochs=100,callbacks=[checkpoint,early])
##Save models
model.save('CoffeDM1.h5')
model.save_weights('CoffeDW.h5')