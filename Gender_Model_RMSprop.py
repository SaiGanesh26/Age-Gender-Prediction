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

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam

print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


data =  pd.read_csv("data/dataset.csv")

data.head()

start = time.time()
try:  
    with tf.device('/device:GPU:7'):
        image_list = []
        for path in data["img_path"]:
            img = cv2.imread("data/" + path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(200,200))
            image_list.append(img)
except RuntimeError as e:
  print(e)
end = time.time()
print("time taken for execution :- {}".format(end-start))

data["image"] = image_list
data.head()

data.info()

#plt.imshow(data["image"][39454])

#normalizing the pixel values
try:  
    with tf.device('/device:GPU:7'):
        x_data = np.array(image_list)/255
        y_data = data["gender"].to_numpy()
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


# # Model

# batch norm SGD Model
chanDim = -1

try:  
    with tf.device('/device:GPU:7'):
        # specify the input size of the images
        images = Input((train_x.shape[1], train_x.shape[2], 1,))
        x = Conv2D(32,kernel_size=(3,3),padding="same")(images)

        x = Activation("relu")(x)

        x= BatchNormalization(axis=chanDim)(x)
        x= MaxPool2D(pool_size=(3,3))(x)
        x= Dropout(0.25)(x)

        x= Conv2D(64, (3,3), padding="same")(x)
        x= Activation("relu")(x)
        x= BatchNormalization(axis=chanDim)(x)
        x= Conv2D(64, (3,3), padding="same")(x)
        x= Activation("relu")(x)
        x= BatchNormalization(axis=chanDim)(x)
        x= MaxPool2D(pool_size=(2,2))(x)
        x= Dropout(0.25)(x)

        x= Conv2D(128, (3,3), padding="same")(x)
        x= Activation("relu")(x)
        x= BatchNormalization(axis=chanDim)(x)

        x= Conv2D(128, (3,3), padding="same")(x)
        x= Activation("relu")(x)
        x= BatchNormalization(axis=chanDim)(x)
        x= MaxPool2D(pool_size=(2,2))(x)
        x= Dropout(0.25)(x)

        x= Flatten()(x)
        
        x= Dense(1024)(x)
        x= Activation("relu")(x)
        x= BatchNormalization(axis=chanDim)(x)
        x= Dropout(0.5)(x)

        x= Dense(1)(x)
        
        predictions = Activation("sigmoid")(x)

        # create the model
        sample_cnn = Model(inputs=images, outputs=predictions)

        # compile the model so that it uses RMSprop 
        sample_cnn.compile(optimizer=RMSprop(), loss="binary_crossentropy", metrics=["acc"])

        print(sample_cnn.summary())

except RuntimeError as e:
  print(e)

start = time.time()

try:  
    with tf.device('/device:GPU:7'):
        # train model
        history = sample_cnn.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=50, batch_size=100, verbose=1)
except RuntimeError as e:
  print(e)
end = time.time()
print("Time spent for training - {}".format(end-start))

try:  
    with tf.device('/device:GPU:7'):
        test_pred = sample_cnn.predict(test_x)
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
print(sample_cnn.evaluate(test_x,test_y))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("accuracy.png")
plt.show()


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig("loss.png")
plt.show()

sample_cnn.save("models/batch_norm_rms")

