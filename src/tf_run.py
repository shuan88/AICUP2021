import tensorflow as tf
import cv2
import numpy as np
import os

def process_an_image(img ,IMAGE_SIZE = (256,256)):
    return cv2.cvtColor(\
                        cv2.resize(img , IMAGE_SIZE),\
                        cv2.COLOR_BGR2RGB)\
                        /255


model_path = "/model/tf_saved_model/inceptionv3_299"

IMAGE_SIZE = (299,299)
# new_model = tf.keras.models.load_model('/content/drive/MyDrive/AICUP2021/saved_model/{}'.format(model_name))
new_model = tf.keras.models.load_model(model_path)
new_model.summary()

class_name = "ok"
dir_name = "/content/station4_white_pin2021-2-23_1614072478/training_data/"
classes = ["ng","ok"]
test_img_dir = dir_name + class_name

print(len(os.listdir(test_img_dir)))
error_count = 0

for img_name in os.listdir(test_img_dir):
    img = cv2.resize(cv2.imread("{}/{}".format(test_img_dir,img_name)) , IMAGE_SIZE)
    img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)/255

    prediction_scores = new_model.predict(np.expand_dims(img, axis=0))
    predicted_index = np.argmax(prediction_scores)
    if predicted_index != (classes.index(class_name)) :
        print(img_name , ":" , prediction_scores)
        error_count += 1
        cv2_imshow(cv2.resize(cv2.imread("{}/{}".format(test_img_dir,img_name)) , IMAGE_SIZE))
print(error_count)