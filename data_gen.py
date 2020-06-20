from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input

DATADIRTRAIN = "output/train"
DATADIRTEST = "output/val"
IMG_SIZE = 300
BATCH_SIZE = 8
datagen = ImageDataGenerator()

#data augmentation for training dataset
data_augmentation = ImageDataGenerator(
    preprocessing_function = preprocess_input,
    rotation_range = 90,
    zoom_range = 0.1,
    horizontal_flip = True,
    vertical_flip = True
)

#reading data from directory dataset and resize it using billinear interpolar
train_generator = data_augmentation.flow_from_directory(
    DATADIRTRAIN,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE
)

# reading dataset for validation
validation_generator = datagen.flow_from_directory(
    DATADIRTEST,
    target_size = (IMG_SIZE, IMG_SIZE),
    batch_size = BATCH_SIZE
)