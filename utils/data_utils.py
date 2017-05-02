import copy, os
import numpy as np
from scipy.io import loadmat
from keras.utils import np_utils

def format_data(dataset):
    (h, w, c) = dataset['image']['shape']
    train_data = dataset['train']['data']
    test_data = dataset['test']['data']

    train_data = train_data.reshape(train_data.shape[0], h, w, c)
    test_data = test_data.reshape(test_data.shape[0], h, w, c)

    dataset['train']['data'] = train_data.astype('float32') / 255
    dataset['test']['data'] = test_data.astype('float32') / 255
    return dataset

def split_data(dataset):
    (w,h,c) = dataset['image']['shape']
    n_classes = dataset['num classes']


    n_valid = dataset['num train examples'] // dataset['validation split']
    n_valid = int(n_valid)
    n_train = dataset['num train examples'] - n_valid
    cutoff = n_valid // n_classes

    train_data = copy.deepcopy(dataset['train']['data'])
    train_labels = copy.deepcopy(dataset['train']['labels'])

    del dataset['train']['data']
    del dataset['train']['labels']

    valid_data = np.zeros((n_valid, h, w, c))
    valid_labels = np.zeros((n_valid, 1))

    label = 0
    val_idx = 0
    remove_list = []
    for idx in range(n_train + n_valid):
        if val_idx >= cutoff:
            label = val_idx // cutoff
        if label >= n_classes:
            break
        if train_labels[idx] == label:
            valid_data[val_idx] = train_data[idx]
            valid_labels[val_idx] = train_labels[idx]
            remove_list.append(idx)
            val_idx += 1
    print(len(remove_list))

    train_data = np.delete(train_data, remove_list, 0)
    train_labels = np.delete(train_labels, remove_list, 0)

    dataset['train']['data'] = train_data
    dataset['train']['labels'] = np_utils.to_categorical(train_labels, n_classes)

    dataset['valid']['data'] = valid_data
    dataset['valid']['labels'] = np_utils.to_categorical(valid_labels, n_classes)

    print(train_data.shape)
    print(train_labels.shape)

    return dataset

def load_data(dataset):
    """Load cifar-3 dataset,

    Args:
        dataset: dictionary defining
            - `io`: a dictionary defining:
                - `relative path`: the relative path to the data directory w.r.t. abs path of `utils.py`
                - `directory`: the directory the data is stored in
                - `extension`: the file format of both the training and testing datasets
                - `train fname`: filename of train dataset
                - `test fname`: filename of test dataset
            - `image`: a dictionary defining:
                - `shape`: (height, width, channels)
                - `unrolled`: height*width*channels
            - `num classes`
            - `num train examples`
            - `num test examples`
            - `validation split`

    Returns:   tuple of train, valid, and test data
        train: dictionary of training data and labels
            - `x`: shape = (,)
            - `y`: shape = (,)
        valid: dictionary of validation data and labels
            - `x`: shape = (,)
            - `y`: shape = (,)
        test: dictionary of data and labels
            - `x`: shape = (,)
            - `y`: None
    """
    dataset['train'] = {}
    dataset['valid'] = {}
    dataset['test'] = {}

    fpath = os.path.join(dataset['io']['relative path'], dataset['io']['directory'])
    train_fpath = os.path.join(fpath, dataset['io']['train fname'] + dataset['io']['extension'])
    test_fpath = os.path.join(fpath, dataset['io']['test fname'] + dataset['io']['extension'])

    n_train = dataset['num train examples'] # 12000
    n_test = dataset['num test examples'] # 3000
    unrolled_len = dataset['image']['unrolled']

    if dataset['io']['extension'] == '.mat':
        train_mat = loadmat(train_fpath)
        test_mat = loadmat(test_fpath)

        dataset['train']['data'] = train_mat.get('data')
        dataset['train']['labels'] = train_mat.get('labels')
        dataset['test']['data'] = test_mat.get('data')

    elif dataset['io']['extension'] == '.bin':
        with np.memmap(train_fpath, dtype='uint8', mode='c', shape=(n_train, unrolled_len+1)) as mm:
            dataset['train']['data'] = mm[np.repeat(np.arange(n_train), unrolled_len), np.tile(np.arange(1,unrolled_len+1), n_train)]
            dataset['train']['labels'] = mm[np.arange(n_train), np.repeat(0, n_train)]

        with np.memmap(test_fpath, dtype='uint8', mode='c', shape=(n_test, unrolled_len)) as mm:
            dataset['test']['data'] = np.reshape(mm, dataset['image']['shape'])

    else:
        raise ValueError, "unsupported filetype: %s \n" %(dataset['io']['extension'])

    dataset = format_data(dataset)
    dataset = split_data(dataset)

    x_train = copy.deepcopy(dataset['train'].pop('data', None))
    y_train = copy.deepcopy(dataset['train'].pop('labels', None))
    del dataset['train']

    x_valid = copy.deepcopy(dataset['valid'].pop('data', None))
    y_valid = copy.deepcopy(dataset['valid'].pop('labels', None))
    del dataset['valid']

    x_test = copy.deepcopy(dataset['test'].pop('data', None))
    del dataset['test']

    train = {'x': x_train, 'y': y_train}
    valid = {'x': x_valid, 'y': y_valid}
    test = {'x': x_test, 'y': None}
    return (train, valid, test)







################################################################################

# np.random.seed(7)
#
# # trainImage contains .label and .data field
# traindata = sio.loadmat('./data_mat/data_batch.mat')
# testdata = sio.loadmat('./data_mat/test_data.mat')
#
# x_train = traindata.get('data')
# x_test = testdata.get('data')
#
# x_train = x_train.reshape(x_train.shape[0], 32, 32,3)
# x_test = x_test.reshape(x_test.shape[0], 32, 32,3)
#
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
#
# x_train /= 255
# x_test /= 255
# input_shape = (32, 32, 3)
# '''
# x_val = x_train[0:1000,:,:,:]
# x_train = x_train[1000:,:,:,:]
# '''
#
# y_train = traindata.get('labels')
#
# x_val = np.zeros((1500, 32, 32, 3))
# y_val = np.zeros((1500, 1))
#
#
# label = 0
# val_idx = 0
# train_idx = 0
# remove_list = []
# for train_idx in range(12000):
#     if val_idx >= 500:
#         label = val_idx/500
#     if label >= 3:
#         break
#     if y_train[train_idx] == label:
#         x_val[val_idx] = x_train[train_idx]
#         y_val[val_idx] = y_train[train_idx]
#         remove_list.append(train_idx)
#         val_idx += 1
# print(len(remove_list))
#
# x_train = np.delete(x_train, remove_list, 0)
# y_train = np.delete(y_train, remove_list, 0)
#
#
# num_classes = 3
# y_train = np_utils.to_categorical(y_train , num_classes)
# y_val = np_utils.to_categorical(y_val , num_classes)
# '''
# y_val = y_train[0:1000,:]
# y_train = y_train[1000:,:]
# '''
# print(x_train.shape)
# print(y_train.shape)
#
# #x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.33, random_state=7)
