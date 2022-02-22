# import tensorflow as tf
import cv2
import numpy as np
import os

print(cv2 .__version__)

model_name = "inceptionv3_2_Fine-tuning"
full_model_path = 'model.onnx'
IMAGE_SIZE = (256,256)
class_name = "ok"
# class_name = "ng"
dir_name = "../照片/New_Data/station4_white_pin2021_5_26_1621999356/training_data/"
classes = ["ng","ok"]

# https://docs.opencv.org/3.4/d6/d0f/group__dnn.html
opencv_net = cv2.dnn.readNetFromONNX(full_model_path)

# full_model_path = 'squeezenet1.0-3.onnx'

""" Single image
image_path = '../照片/station4_white_pin2021-2-9_1612936555/testing_data/ok/station4_10pin_pin_001.jpg'
img = cv2.imread(image_path)
img = cv2.resize(cv2.imread(image_path) , IMAGE_SIZE)
img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
print(np.expand_dims(img, axis=0).shape)
# print(img)
opencv_net.setInput(np.expand_dims(img, axis=0))
out = opencv_net.forward()
print(out)
"""

test_img_dir = dir_name + class_name
print(len(os.listdir(test_img_dir)))
error_count = 0
for img_name in os.listdir(test_img_dir):
    try:
        # img = cv2.resize(cv2.imread("{}/{}".format(test_img_dir,img_name)) , IMAGE_SIZE)
        img = cv2.resize(cv2.imread("{}/{}".format(test_img_dir,img_name)) , IMAGE_SIZE)
        img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255
        opencv_net.setInput(np.expand_dims(img, axis=0))
        prediction_scores = opencv_net.forward()
        # prediction_scores = new_model.predict(np.expand_dims(img, axis=0))
        predicted_index = np.argmax(prediction_scores)
        # if predicted_index != 1:
        if predicted_index != (classes.index(class_name)) :
            # print(img_name , ":" , prediction_scores)
            error_count += 1
            cv2.imshow("{},{}".format(error_count,prediction_scores) ,\
                cv2.resize(cv2.imread("{}/{}".format(test_img_dir,img_name)) , IMAGE_SIZE))
            cv2.waitKey(1)
    except:
        os.remove("{}/{}".format(test_img_dir,img_name))
        # continue
print(error_count)
print("predicted score : " , 1. -(error_count/len(os.listdir(test_img_dir))) )
cv2.destroyAllWindows()