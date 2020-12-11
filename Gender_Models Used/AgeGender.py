#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/SaiGanesh26/Gender-Classification-Age-Prediction/blob/master/AgeGender.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Age Prediction & Gender Classification

# In[1]:


#!pip install cvlib


# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import cvlib as cv
import tensorflow as tf
from tensorflow.keras.models import load_model
# from google.colab import  files
# from google.colab.patches import cv2_imshow


# #### Using OpenCV detect face for face detection 

# In[6]:


#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# #### Video Input
# * If existing video file is to be used then file path is given
# * If live web cam is used just enter 0

# In[ ]:


gen_model = load_model("models/batch_norm_rms_93acc_50epo")


# In[4]:


def detect_face_video(img):
    label_dict={0:'Female',1:'Male'}
    color_dict={0:(0,255,0),1:(0,0,255)}
    #Using OpenCV detect face for face detection available in cvlib library
    faces, confidences = cv.detect_face(img) 
#     face_coord = face_cascade.detectMultiScale(img,scaleFactor = 1.2,minNeighbors = 5)  # This gives us the coordinates of rectangle drawn across the face 
#     for x,y,w,h in face_coord:
    for index,face in enumerate(faces):
        x1,y1 = face[0],face[1]
        x2,y2 = face[2],face[3]
        image = img[y1:y2,x1:x2]
        image_gry = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        resize_img = cv2.resize(image_gry,(200,200))
        norm_img = resize_img/255.0
        reshaped=np.reshape(norm_img,(1,200,200,1))
        #reshaped = np.vstack([reshaped])

        try:  
            with tf.device('/device:GPU:5'):
                gender_res = gen_model.predict(reshaped)
        except RuntimeError as e:
            print(e)
        if(gender_res[0][0]<0.5):
            label=0
        else:
            label=1
     
        cv2.rectangle(img,(x1,y1),(x2,y2),color_dict[label],thickness = 5)
#         cv2.rectangle(img,(x1,y1),(x2,y2),color_dict[label],)
        cv2.putText(img, label_dict[label], (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    return img


# ## For image

# In[9]:


# def detect_face_image(img):
#     label_dict={0:'Female',1:'Male'}
#     color_dict={0:(0,255,0),1:(0,0,255)}
#     faces, confidences = cv.detect_face(img) 
# #     face_coord = face_cascade.detectMultiScale(img,scaleFactor = 1.2,minNeighbors = 5)  # This gives us the coordinates of rectangle drawn across the face 
# #     for x,y,w,h in face_coord:
#     for index,face in enumerate(faces):
#         x1,y1 = face[0],face[1]
#         x2,y2 = face[2],face[3]
#         image = img[y1:y2,x1:x2]
#         image_gry = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#         resize_img = cv2.resize(image_gry,(200,200))
#         norm_img = resize_img/255.0
#         reshaped=np.reshape(norm_img,(1,200,200,1))
#         #reshaped = np.vstack([reshaped])

#         try:  
#             with tf.device('/device:GPU:5'):
#                 gender_res = gen_model.predict(reshaped)
#         except RuntimeError as e:
#             print(e)
#         if(gender_res[0][0]<0.5):
#             label=0
#         else:
#             label=1
     
#         cv2.rectangle(img,(x1,y1),(x2,y2),color_dict[label],thickness = 5)
# #         cv2.rectangle(img,(x1,y1),(x2,y2),color_dict[label],)
#         cv2.putText(img, label_dict[label], (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
#     return img


# # In[14]:


# image = cv2.imread("temp1.JPG")
# out_image = detect_face_image(image)
# plt.imshow(out_image)


# ## For Existing video

# In[2]:


video = cv2.VideoCapture('test1.mp4')  # Sincle colab has no option for live recording, existing file is used


# In[3]:


width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('test5_out.mp4',cv2.VideoWriter_fourcc(*'XVID'),25,(width,height))


# * colab gives the output of video files as a series of each frame present in video file

# In[ ]:


if video.isOpened() == False:
  print('Error! File Not found')
while video.isOpened():
    ret,frame = video.read()
    if ret == True:
        frame = detect_face_video(frame)  #detect the face for each frame displayed on video
        out.write(frame)
    if cv2.waitKey(27) & 0xFF == ord('q'): #if live webcam is given input then video can be quit using letter 'q'
        break
    else:
        break
video.release()
out.release()
cv2.destroyAllWindows()


# ## For live Webcam Video

# In[5]:


# webcam = cv2.VideoCapture(0)
# while True:
#     ret,frame = webcam.read()
#     frame = detect_face_video(frame)  #detect the face for each frame displayed on video
#     cv2.imshow("gender detection", frame)
#     k = cv2.waitKey(30) & 0xff
#     if k==27:
#         break
# webcam.release()
# cv2.destroyAllWindows()

