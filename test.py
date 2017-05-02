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

import csv, time
import numpy as np
import scipy.io as sio
#from scipy.io import loadmat
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from keras.utils import np_utils

# trainImage contains .label and .data field
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
    axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')

    axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.draw()
    plt.savefig('plot')

np.random.seed(7)

# trainImage contains .label and .data field
traindata = sio.loadmat('./data_mat/data_batch.mat')
testdata = sio.loadmat('./data_mat/test_data.mat')

x_train = traindata.get('data')
x_test = testdata.get('data')

x_train = x_train.reshape(x_train.shape[0], 32, 32,3)
x_test = x_test.reshape(x_test.shape[0], 32, 32,3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255
input_shape = (32, 32, 3)
'''
x_val = x_train[0:1000,:,:,:]
x_train = x_train[1000:,:,:,:]
'''

y_train = traindata.get('labels')

x_val = np.zeros((1500, 32, 32, 3))
y_val = np.zeros((1500, 1))


label = 0
val_idx = 0
train_idx = 0
remove_list = []
for train_idx in range(12000):
    if val_idx >= 500:
        label = val_idx/500
    if label >= 3:
        break
    if y_train[train_idx] == label:
        x_val[val_idx] = x_train[train_idx]
        y_val[val_idx] = y_train[train_idx]
        remove_list.append(train_idx)
        val_idx += 1
print(len(remove_list))

x_train = np.delete(x_train, remove_list, 0)
y_train = np.delete(y_train, remove_list, 0)


num_classes = 3
y_train = np_utils.to_categorical(y_train , num_classes)
y_val = np_utils.to_categorical(y_val , num_classes)
'''
y_val = y_train[0:1000,:]
y_train = y_train[1000:,:]
'''
print(x_train.shape)
print(y_train.shape)

#x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=7)




data_augmentation = False
nb_classes = 3
'''
model = Sequential()

model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# model.add(Conv2D(FEATURES_2, 2,2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
'''
'''
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
'''
'''
batch_size = 32 # in each iteration, we consider 32 training examples at once
num_epochs = 200 # we iterate 200 times over the entire training set
kernel_size = 3 # we will use 3x3 kernels throughout
pool_size = 2 # we will use 2x2 pooling throughout
conv_depth_1 = 32 # we will initially have 32 kernels per conv. layer...
conv_depth_2 = 64 # ...switching to 64 after the first pooling layer
drop_prob_1 = 0.25 # dropout after pooling with probability 0.25
drop_prob_2 = 0.5 # dropout in the FC layer with probability 0.5
hidden_size = 512 # the FC layer will have 512 neurons

num_train, height, width, depth = x_train.shape
inp = Input(shape=(height, width, depth))
conv_1 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
conv_2 = Conv2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)
drop_1 = Dropout(drop_prob_1)(pool_1)
# Conv [64] -> Conv [64] -> Pool (with dropout on the pooling layer)
conv_3 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
conv_4 = Conv2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)
drop_2 = Dropout(drop_prob_1)(pool_2)
# Now flatten to 1D, apply FC -> ReLU (with dropout) -> softmax
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
out = Dense(num_classes, activation='softmax')(drop_3)

model = Model(inputs=inp, outputs=out) # To define a model, just specify its input and output layers

model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy


'''
'''
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))


st = time.time()
# opt = optimizers.rmsprop(lr=0.001, decay=1e-6)
epo = 300
lrate = 0.01
decay = lrate/epo
batch=32
sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='categorical_crossentropy',
              #optimizer=sgd,
              optimizer='adam',
              metrics=['accuracy'])
'''

'''
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')
x_train /= 255
x_val /= 255'''
epo = 50
batch=64
model = Sequential()
'''
#1
model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(32,32,3)))
model.add(Conv2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
# model.add(Conv2D(FEATURES_2, 2,2, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))
'''


#2
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
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# Compile model
lrate = 0.01
decay = lrate/epo
sgd = SGD(lr=lrate, momentum=0.9, decay=1e-6, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


if not data_augmentation:
    print('Not using data augmentation.')
    res = model.fit(x_train, y_train,
              batch_size=batch,
              epochs=epo,
              validation_data=(x_val, y_val),
              shuffle=True)
    #model.fit(x_train, y_train, epochs=epo, batch_size=batch,validation_data = (x_val,y_val))
    score = model.evaluate(x_train, y_train)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    res = model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch),
                        steps_per_epoch=x_train.shape[0] // batch,
                        epochs=epo,
                        validation_data=(x_val, y_val))


y_test = model.predict_classes(x_test, verbose=0)
print(y_test)
y_test = np.array(y_test)

with open('out_1.csv','wb') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for val in y_test:
        wr.writerow([val])

plot_model_history(res)
