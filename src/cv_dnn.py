# import tensorflow as tf
import cv2
import numpy as np
import os

model_name = "inceptionv3_2_Fine-tuning"
IMAGE_SIZE = (299,299)

# model_path ='../model/{}/'.format(model_name)
# model_path ='../model/inceptionv3_2_Fine-tuning/saved_model.pb'
model_path ='../model/inceptionv3_2_Fine-tuning/'
image_path = '../照片/station4_white_pin2021-2-9_1612936555/testing_data/ok/station4_10pin_pin_001.jpg'

# tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'graph.pbtxt')
# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
tensorflowNet = cv2.dnn.readNetFromTensorflow(model_path)

# Input image
# img = cv2.resize(cv2.imread('/content/station4_white_pin2021-2-9_1612936555/testing_data/ok/station4_10pin_pin_001.jpg') , IMAGE_SIZE)
img = cv2.imread(image_path)

# blob = cv2.dnn.blobFromImage(img,1./255,size=(IMAGE_SIZE),swapRB=True)
# # images_	=	cv2.dnn.imagesFromBlob(blob,img)
# print(blob.shape)
# tensorflowNet.setInput(blob)
# out = tensorflowNet.forward()
# print(out)

img = cv2.resize(cv2.imread(image_path) , IMAGE_SIZE)
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
print(np.expand_dims(img, axis=0).shape)
# print(img)
tensorflowNet.setInput(np.expand_dims(img, axis=0))
out = tensorflowNet.forward()
print(out)