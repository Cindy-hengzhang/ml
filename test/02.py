import csv, time
import numpy as np
from ..utils import dataset_utils, data_utils, plot_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

#@TODO: rewrite keras model somewhat from stratch
# as per, piazza @2325

# we must cite https://github.com/fchollet/keras/blob/master/examples/cifar10_cnn.py,
# as this code seems to be originaly based off of it
# but we need to remove references to http://parneetk.github.io/blog/cnn-cifar10/
# as that violates the polity below

'''
    For computer vision task of homework 9, please consider the following as homework course policies:

    1. You are NOT allowed to use any pre-trained neural network weights for solving the task.

    2. You are only allowed to use standard examples of neural networks from Tensorflow/Keras/Caffe/Torch (from the original documentation repository). You are NOT allowed to use other private/public Github repository codes for the task.

    3. If you choose to use one of the standard neural network examples from these frameworks, you have to try different modifications of the network, identify and analyze the impact of adding / modifying / deleting certain layers, tuning hyper-parameters, etc and report the results in the write-up. Using the model and the parameters as is and submitting only those results will not fetch full points.

    4. If you choose to use one of the standard neural network examples from these frameworks, you have to CITE the code repository in the write-up, indicating how you used the code. Any public code used without citation in the write-up will be considered plagiarism.
'''

dataset = dataset_utils.get_dataset(filetype='.mat')
(train, valid, test) = data_utils.load_data(dataset)

data_augmentation = True
num_classes = 3

model = Sequential()

model.add(ZeroPadding2D(padding=(1,1), input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(1024, (3,3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Conv2D(1024, (1,1)))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Conv2D(1024, (3,3)))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# from keras.utils.layer_utils import print_summary
# print_summary(model)

# Compile model
epochs = 50
lrate = 0.01
decay = lrate/epochs
batch_size=64
sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])


if not data_augmentation:
    print('Not using data augmentation.')
    res = model.fit(train['x'], train['y'],
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(valid['x'], valid['y']),
              shuffle=True,
              verbose=2)
    #model.fit(train['x'], train['y'], epochs=epochs, batch_size=batch,validation_data = (valid['x'], valid['y']))
    score = model.evaluate(train['x'], train['y'])
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    train_datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zca_whitening=True,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(zca_whitening=True)
    test_datagen.fit(valid['x'])
    validation_generator = test_datagen.flow(valid['x'], valid['y'])

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    train_datagen.fit(train['x'])
    train_generator = train_datagen.flow(train['x'], train['y'], batch_size=batch_size)

    # Fit the model on the batches generated by datagen.flow().
    res = model.fit_generator(
        train_generator,
        steps_per_epoch=train['x'].shape[0] // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=400,
        shuffle=True,
        verbose=2)

test['y_pred'] = model.predict_classes(test['x'], verbose=0)
print(test['y_pred'])

with open('out_1.csv','wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for pred in test['y_pred']:
        wr.writerow([pred])

plot_utils.plot_model_history(res)
