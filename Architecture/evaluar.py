import os
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.python.keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
modelo= './Model/CoffeDM.h5' #Modelo
pesos='./Model/CoffeDW.h5'
data_validacion='./Image/val'
altura,longitud=224,224
cnn = load_model(modelo)
cnn.load_weights(pesos)
K.clear_session()
validacion_data = ImageDataGenerator(
)
imagen_validacion=validacion_data.flow_from_directory(
    data_validacion,
    target_size=(altura,longitud),
    class_mode='categorical'
)
y_pred= cnn.predict(imagen_validacion)
y_pred= np.argmax(y_pred,axis=1)
print('matriz de confusion')
print(imagen_validacion.classes)
cm=confusion_matrix(imagen_validacion.classes,y_pred)
print("Report de clasificacion")
target_names = ['Coffe','Other']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
print(classification_report(imagen_validacion.classes,y_pred,target_names=target_names))
