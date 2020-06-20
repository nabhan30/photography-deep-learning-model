import cv2
import tensorflow as tf
import os
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CATEGORIES = ["astronomy", "baby", "landscape", "macro", "sport", "wedding", "wildlife"]

def prepare(file):
    IMG_SIZE = 300
    img_array = cv2.imread(file)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return np.reshape(new_array, [1, IMG_SIZE, IMG_SIZE, 3])

model = tf.keras.models.load_model("model/ResNet50_model_weights.h5")
image = prepare("data_new_input/nature2.jpg")
prediction = model.predict(image)
prediction = list(prediction[0])

print(CATEGORIES[prediction.index(max(prediction))])