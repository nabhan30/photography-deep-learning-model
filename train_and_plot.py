import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as pyplot
import finetune_model
import data_gen

NUM_EPOCHS = 50
BATCH_SIZE = 8
num_train_images = 2235
finetune_model = finetune_model.finetune_model
train_generator = data_gen.train_generator
validation_generator = data_gen.validation_generator

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="model/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(
        train_generator,
        epochs=NUM_EPOCHS,
        workers=8,
        validation_data=validation_generator,
        steps_per_epoch=num_train_images // BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks_list
    )

# Plot the training and validation loss + accuracy
print(history.history.keys())
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

pyplot.plot(epochs, acc, 'r.')
pyplot.plot(epochs, val_acc, 'r')
pyplot.title('Training and validation accuracy')

pyplot.figure()
pyplot.plot(epochs, loss, 'r.')
pyplot.plot(epochs, val_loss, 'r-')
pyplot.title('Training and validation loss')
pyplot.show()

pyplot.savefig('acc_vs_epochs.png')