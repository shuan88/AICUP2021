# import tensorflow as tf
import cv2
import numpy as np
import os

print(cv2 .__version__)

model_name = "inceptionv3_2_Fine-tuning"
IMAGE_SIZE = (256,256)

full_model_path = 'model.onnx'
image_path = '../照片/station4_white_pin2021-2-9_1612936555/testing_data/ok/station4_10pin_pin_001.jpg'

# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
opencv_net = cv2.dnn.readNetFromONNX(full_model_path)
# opencv_net = cv2.dnn.readNetFromONNX('squeezenet1.0-3.onnx')

img = cv2.imread(image_path)

img = cv2.resize(cv2.imread(image_path) , IMAGE_SIZE)
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
print(np.expand_dims(img, axis=0).shape)
# print(img)
opencv_net.setInput(np.expand_dims(img, axis=0))
out = opencv_net.forward()
print(out)