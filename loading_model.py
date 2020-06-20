from keras.applications.resnet50 import ResNet50, preprocess_input
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

HEIGHT = 300
WIDTH = 300

base_model = ResNet50(weights='imagenet', 
                      include_top=False,
                      input_shape=(HEIGHT, WIDTH, 3))