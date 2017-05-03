import csv, time
import numpy as np
from ..utils import dataset_utils, data_utils, plot_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD

dataset = dataset_utils.get_dataset(filetype='.mat')
(train, valid, test) = data_utils.load_data(dataset)

num_classes = 3

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# from keras.utils.layer_utils import print_summary
# print_summary(model)

st = time.time()
# opt = optimizers.rmsprop(lr=0.001, decay=1e-6)
epochs = 200
lrate = 0.001
decay = lrate / epochs
batch_size = 64
sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train['x'])

# Fit the model on the batches generated by datagen.flow().
info = model.fit_generator(datagen.flow(train['x'], train['y'],
                                        batch_size=batch_size),
                           steps_per_epoch=train['x'].shape[0] // batch_size,
                           epochs=epochs,
                           validation_data=(valid['x'], valid['y']),
                           verbose=2)

# info = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
# score = model.evaluate(x_train, y_train)

test['y_pred'] = model.predict_classes(test['x'], verbose=0)
print(test['y_pred'])
with open('out.csv', 'wb') as myfile:
    out = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for pred in test['y_pred']:
        out.writerow([pred])
