import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
sns.set_style("whitegrid")

#@TODO: rewrite the function from stratch
# as per, piazza @2325

'''
    For computer vision task of homework 9, please consider the following as homework course policies:

    1. You are NOT allowed to use any pre-trained neural network weights for solving the task.

    2. You are only allowed to use standard examples of neural networks from Tensorflow/Keras/Caffe/Torch (from the original documentation repository). You are NOT allowed to use other private/public Github repository codes for the task.

    3. If you choose to use one of the standard neural network examples from these frameworks, you have to try different modifications of the network, identify and analyze the impact of adding / modifying / deleting certain layers, tuning hyper-parameters, etc and report the results in the write-up. Using the model and the parameters as is and submitting only those results will not fetch full points.

    4. If you choose to use one of the standard neural network examples from these frameworks, you have to CITE the code repository in the write-up, indicating how you used the code. Any public code used without citation in the write-up will be considered plagiarism.
'''

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


# # trainImage contains .label and .data field
# def plot_model_history(model_history):
#     fig, axs = plt.subplots(1,2,figsize=(15,5))
#     # summarize history for accuracy
#     axs[0].plot(range(1,len(model_history.history['acc'])+1),model_history.history['acc'])
#     axs[0].plot(range(1,len(model_history.history['val_acc'])+1),model_history.history['val_acc'])
#     axs[0].set_title('Model Accuracy')
#     axs[0].set_ylabel('Accuracy')
#     axs[0].set_xlabel('Epoch')
#
#     axs[0].set_xticks(np.arange(1,len(model_history.history['acc'])+1),len(model_history.history['acc'])/10)
#     axs[0].legend(['train', 'val'], loc='best')
#     # summarize history for loss
#     axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
#     axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
#     axs[1].set_title('Model Loss')
#     axs[1].set_ylabel('Loss')
#     axs[1].set_xlabel('Epoch')
#     axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
#     axs[1].legend(['train', 'val'], loc='best')
#     plt.draw()
#     plt.savefig('plot')
