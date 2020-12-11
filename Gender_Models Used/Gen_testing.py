#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np
import pandas as pd 
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
#%matplotlib inline
from tensorflow.keras.models import load_model

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


wiki_process =  pd.read_csv("/home/students/saivuppa/Project/data/dataset.csv")

wiki_process.head()

start = time.time()
try:  
    with tf.device('/device:GPU:7'):
        image_list = []
        for path in wiki_process["img_path"]:
            img = cv2.imread("/home/students/saivuppa/Project/data/" + path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(200,200))
            image_list.append(img)
except RuntimeError as e:
  print(e)
end = time.time()
print("time taken for execution :- {}".format(end-start))

wiki_process["image"] = image_list
wiki_process.head()

wiki_process.info()

#plt.imshow(wiki_process["image"][39454])

#normalizing the pixel values
try:  
    with tf.device('/device:GPU:7'):
        x_data = np.array(image_list)/255
        y_data = wiki_process["gender"].to_numpy()
except RuntimeError as e:
  print(e)

x_data.shape

y_data.shape

# image_x will contain the original grayscale images 
x_data = x_data.reshape((x_data.shape[0],200,200,1))

print("x_data shape: {}".format(x_data.shape))
print("y_data shape: {}".format(y_data.shape))

train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.33, random_state=42)

print("train_x shape: {}".format(train_x.shape))
print("train_y shape: {}\n".format(train_y.shape))

print("test_x shape: {}".format(test_x.shape))
print("test_y shape: {}".format(test_y.shape))


gen_model = load_model("/home/students/saivuppa/Project/models/batch_norm_rms")

try:  
    with tf.device('/device:GPU:7'):
        test_pred = gen_model.predict(test_x)
        for i in test_pred:
            if i[0] >= 0.5:
                i[0] = 1
            else:
                i[0] = 0
        print(test_pred)
except RuntimeError as e:
  print(e)

print(classification_report(test_y,test_pred))
print(confusion_matrix(test_y,test_pred))
print(gen_model.evaluate(test_x,test_y))

