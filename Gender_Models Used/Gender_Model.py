#!/usr/bin/env python
# coding: utf-8

# In[24]:


import cv2
import numpy as np
import pandas as pd 
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
#get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report,confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam


# In[32]:


print("Num GPUs Available: ", tf.config.experimental.list_physical_devices('GPU'))


# In[26]:


#pip install numba 


# In[27]:



# In[28]:


wiki_process =  pd.read_csv("data/dataset.csv")


# In[29]:


wiki_process.head()


# In[31]:


start = time.time()
try:  
    with tf.device('/device:GPU:3'):
        image_list = []
        for path in wiki_process["img_path"]:
            img = cv2.imread("data/" + path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img,(200,200))
            image_list.append(img)
except RuntimeError as e:
  print(e)
end = time.time()
print("time taken for execution :- {}".format(end-start))


# In[6]:


# wiki_process = wiki_process.head(500)
wiki_process["image"] = image_list
wiki_process.head()


# In[7]:


wiki_process.info()


# In[8]:


plt.imshow(wiki_process["image"][39454])


# In[9]:


#normalizing the pixel values
try:  
    with tf.device('/device:GPU:3'):
        x_data = np.array(image_list)/255
        y_data = wiki_process["gender"].to_numpy()
except RuntimeError as e:
  print(e)


# In[10]:


x_data.shape


# In[11]:


y_data.shape


# In[12]:


# image_x will contain the original grayscale images 
x_data = x_data.reshape((x_data.shape[0],200,200,1))

print("x_data shape: {}".format(x_data.shape))
print("y_data shape: {}".format(y_data.shape))


# In[13]:


train_x, test_x, train_y, test_y = train_test_split(x_data, y_data, test_size=0.3, random_state=42)

print("train_x shape: {}".format(train_x.shape))
print("train_y shape: {}\n".format(train_y.shape))

print("test_x shape: {}".format(test_x.shape))
print("test_y shape: {}".format(test_y.shape))


# In[14]:


# num_subjects = np.unique(y_data).shape[0]
# print("Number of subjects: {}".format(np.unique(y_data).shape[0]))

# if tf.config.experimental.list_physical_devices('GPU'):
#     strategy = tf.distribute.MirroredStrategy()
# else:  # use default strategy
#     strategy = tf.distribute.get_strategy() 
# print(strategy)
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

        # create the model using the layers we defined previously
        sample_cnn = Model(inputs=images, outputs=predictions)

        # compile the model so that it uses Adam for optimization during training with cross-entropy loss
        sample_cnn.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["acc"])

        # print out a summary of the model achitecture
        print(sample_cnn.summary())

except RuntimeError as e:
  print(e)

# In[40]:


start = time.time()
# class_weights = compute_class_weight("balanced", np.unique(train_y), train_y)
# class_weights = dict(enumerate(class_weights))
# try:  
#     with strategy.scope():
#         # train model
history = sample_cnn.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=1,batch_size=50, verbose=1)
# except RuntimeError as e:
#   print(e)
end = time.time()
print("Time spent for training - {}".format(end-start))


# In[41]:


# try:  
#     with tf.device('/device:GPU:3'):
test_pred = sample_cnn.predict(test_x)
for i in test_pred:
    if i[0] >= 0.5:
        i[0] = 1
    else:
        i[0] = 0
print(test_pred)
# except RuntimeError as e:
#   print(e)


# In[42]:


test_pred[test_pred<0.5]


# In[43]:


print(classification_report(test_y,test_pred))
print(confusion_matrix(test_y,test_pred))


# In[52]:


test_y[0]


# In[30]:


# for i in test_pred:
#     if i[0] >= 0.5:
#         i[0] = 1
#     else:
#         i[0] = 0
# test_pred


# In[44]:


history.history.keys()


# In[47]:


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[63]:


sample_cnn.save("models/adam_model")


# In[ ]:




