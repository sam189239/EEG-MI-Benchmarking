## IMPORTS
import mne 
import numpy as np
import pandas as pd
import torch
import os
import braindecode
import matplotlib.pyplot as plt
from braindecode.models import EEGNetv4, EEGConformer, ATCNet, EEGITNet, EEGInception
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import create_windows_from_events
from braindecode.preprocessing import exponential_moving_standardize, preprocess, Preprocessor

import scipy.io
import pywt
from mne.decoding import CSP # Common Spatial Pattern Filtering
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Reshape, ConvLSTM1D, Conv2D
from keras import regularizers
from sklearn import preprocessing
from sklearn.metrics import *
from sklearn.model_selection import ShuffleSplit


## FEATURE EXTRACTION

def wpd(X): 
    coeffs = pywt.WaveletPacket(X,'db4',mode='symmetric',maxlevel=5)
    return coeffs
             
def feature_bands(x):
    
    Bands = np.empty((8,x.shape[0],x.shape[1],30)) # 8 freq band coefficients are chosen from the range 4-32Hz
    
    for i in range(x.shape[0]):
        for ii in range(x.shape[1]):
             pos = []
             C = wpd(x[i,ii,:]) 
             pos = np.append(pos,[node.path for node in C.get_level(5, 'natural')])
             for b in range(1,9):
                 Bands[b-1,i,ii,:] = C[pos[b]].data
        
    return Bands

## MODELS

def build_mlp_classifier(num_layers = 1, lr = 0.01):
    classifier = Sequential()
    classifier.add(Flatten())
    #First Layer
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', 
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.2))
    # Intermediate Layers
    for itr in range(num_layers):
        classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', 
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.2))   
    # Last Layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = keras.optimizers.Adam(lr=lr) , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


def build_cnn_classifier(input_shape, num_layers=1, lr = 0.01):
    classifier = Sequential()
    
    # First Convolutional Layer
    classifier.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    classifier.add(MaxPooling1D(pool_size=2))
    
    # Intermediate Convolutional Layers
    for _ in range(num_layers):
        classifier.add(Conv1D(32, kernel_size=3, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=2))
    
    # Flattening Layer
    classifier.add(Flatten())
    
    # Fully Connected Layers
    classifier.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    classifier.add(Dropout(0.3))
    
    # Output Layer
    classifier.add(Dense(units=4, activation='softmax'))

    # Compiling the model
    classifier.compile(optimizer=keras.optimizers.Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

def build_cnn2d_classifier(input_shape, num_layers=1):
    classifier = Sequential()
    
    # First Convolutional Layer
    classifier.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=input_shape))
    classifier.add(Conv2D(64, kernel_size=3, activation='relu'))
    classifier.add(MaxPooling2D(pool_size=2))
    
    # Intermediate Convolutional Layers
    for _ in range(num_layers):
        classifier.add(Conv2D(32, kernel_size=3, activation='relu'))
        classifier.add(MaxPooling2D(pool_size=2))
    
    # Flattening Layer
    classifier.add(Flatten())
    
    # Fully Connected Layers
    classifier.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    classifier.add(Dropout(0.5))
    
    # Output Layer
    classifier.add(Dense(units=4, activation='softmax'))

    # Compiling the model
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

def build_convlstm_classifier(input_shape, num_layers=1):
    classifier = Sequential()
    
    # First ConvLSTM1D Layer
    classifier.add(Reshape((-1, 64, 1), input_shape=input_shape)) ## TODO: generalize reshape
    # classifier.add(ConvLSTM2D(32, kernel_size=(3, 1), activation='relu', input_shape=(None, 64)))
    
    # Intermediate ConvLSTM1D Layers
    for _ in range(num_layers):
        classifier.add(ConvLSTM1D(64, kernel_size=3, activation='relu'))
    
    # Flattening Layer
    classifier.add(Flatten())
    
    # Fully Connected Layers
    classifier.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    classifier.add(Dropout(0.5))
    
    # Output Layer
    classifier.add(Dense(units=4, activation='softmax'))

    # Compiling the model
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return classifier

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Permute, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

def build_eegnet_classifier(nb_classes=4, Chans = 22, Samples = 751, 
             dropoutRate = 0.3, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = Input(shape = (Chans, Samples, 1))

    ##################################################################
    block1       = Conv2D(F1, (1, kernLength), padding = 'same',
                                   input_shape = (Chans, Samples, 1),
                                   use_bias = False)(input1)
    block1       = BatchNormalization()(block1)
    block1       = DepthwiseConv2D((Chans, 1), use_bias = False, 
                                   depth_multiplier = D,
                                   depthwise_constraint = max_norm(1.))(block1)
    block1       = BatchNormalization()(block1)
    block1       = Activation('elu')(block1)
    block1       = AveragePooling2D((1, 4))(block1)
    block1       = dropoutType(dropoutRate)(block1)
    
    block2       = SeparableConv2D(F2, (1, 16),
                                   use_bias = False, padding = 'same')(block1)
    block2       = BatchNormalization()(block2)
    block2       = Activation('elu')(block2)
    block2       = AveragePooling2D((1, 8))(block2)
    block2       = dropoutType(dropoutRate)(block2)
        
    flatten      = Flatten(name = 'flatten')(block2)
    
    dense        = Dense(nb_classes, name = 'dense', 
                         kernel_constraint = max_norm(norm_rate))(flatten)
    softmax      = Activation('softmax', name = 'softmax')(dense)
    classifier = Model(inputs=input1, outputs=softmax)
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

def square(x):
    return K.square(x)

def log(x):
    return K.log(K.clip(x, min_value = 1e-7, max_value = 10000)) 

def build_shallow_conv_net(nb_classes=4, Chans = 22, Samples = 751, dropoutRate = 0.5):
    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(40, (1, 13), 
                        input_shape=(Chans, Samples, 1),
                        kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(40, (Chans, 1), use_bias=False, 
                          kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation(square)(block1)
    block1       = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1       = Activation(log)(block1)
    block1       = Dropout(dropoutRate)(block1)
    flatten      = Flatten()(block1)
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    return Model(inputs=input_main, outputs=softmax)

def build_deep_conv_net(nb_classes=4, Chans = 22, Samples = 751, dropoutRate = 0.5):
    # start the model
    input_main   = Input((Chans, Samples, 1))
    block1       = Conv2D(25, (1, 5), 
                                 input_shape=(Chans, Samples, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(input_main)
    block1       = Conv2D(25, (Chans, 1),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block1       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1       = Activation('elu')(block1)
    block1       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1       = Dropout(dropoutRate)(block1)
  
    block2       = Conv2D(50, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block1)
    block2       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block2)
    block2       = Activation('elu')(block2)
    block2       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2       = Dropout(dropoutRate)(block2)
    
    block3       = Conv2D(100, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block2)
    block3       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block3)
    block3       = Activation('elu')(block3)
    block3       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3       = Dropout(dropoutRate)(block3)
    
    block4       = Conv2D(200, (1, 5),
                                 kernel_constraint = max_norm(2., axis=(0,1,2)))(block3)
    block4       = BatchNormalization(epsilon=1e-05, momentum=0.9)(block4)
    block4       = Activation('elu')(block4)
    block4       = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4       = Dropout(dropoutRate)(block4)
    
    flatten      = Flatten()(block4)
    
    dense        = Dense(nb_classes, kernel_constraint = max_norm(0.5))(flatten)
    softmax      = Activation('softmax')(dense)
    
    return Model(inputs=input_main, outputs=softmax)

from tensorflow.keras.layers import concatenate


def build_eegnet_fusion(nb_classes, Chans=64, Samples=128,
                  dropoutRate=0.5, norm_rate=0.25, dropoutType='Dropout', cpu=False):
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    if cpu:
        input_shape = (Samples, Chans, 1)
        conv_filters = (64, 1)
        conv_filters2 = (96, 1)
        conv_filters3 = (128, 1)

        depth_filters = (1, Chans)
        pool_size = (4, 1)
        pool_size2 = (8, 1)
        separable_filters = (8, 1)
        separable_filters2 = (16, 1)
        separable_filters3 = (32, 1)

        axis = -1
    else:
        input_shape = (1, Chans, Samples)
        conv_filters = (1, 64)
        conv_filters2 = (1, 96)
        conv_filters3 = (1, 128)

        depth_filters = (Chans, 1)
        pool_size = (1, 4)
        pool_size2 = (1, 8)
        separable_filters = (1, 8)
        separable_filters2 = (1, 16)
        separable_filters3 = (1, 32)

        axis = 1

    F1 = 8
    F1_2 = 16
    F1_3 = 32
    F2 = 16
    F2_2 = 32
    F2_3 = 64
    D = 2
    D2 = 2
    D3 = 2

    input1 = Input(shape=input_shape)
    block1 = Conv2D(F1, conv_filters, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.))(block1)
    block1 = BatchNormalization(axis=axis)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D(pool_size)(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, separable_filters,
                             use_bias=False, padding='same')(block1)  # 8
    block2 = BatchNormalization(axis=axis)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D(pool_size2)(block2)
    block2 = dropoutType(dropoutRate)(block2)
    block2 = Flatten()(block2)  # 13

    # 8 - 13

    input2 = Input(shape=input_shape)
    block3 = Conv2D(F1_2, conv_filters2, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input2)
    block3 = BatchNormalization(axis=axis)(block3)
    block3 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D2,
                             depthwise_constraint=max_norm(1.))(block3)
    block3 = BatchNormalization(axis=axis)(block3)
    block3 = Activation('elu')(block3)
    block3 = AveragePooling2D(pool_size)(block3)
    block3 = dropoutType(dropoutRate)(block3)

    block4 = SeparableConv2D(F2_2, separable_filters2,
                             use_bias=False, padding='same')(block3)  # 22
    block4 = BatchNormalization(axis=axis)(block4)
    block4 = Activation('elu')(block4)
    block4 = AveragePooling2D(pool_size2)(block4)
    block4 = dropoutType(dropoutRate)(block4)
    block4 = Flatten()(block4)  # 27
    # 22 - 27

    input3 = Input(shape=input_shape)
    block5 = Conv2D(F1_3, conv_filters3, padding='same',
                    input_shape=input_shape,
                    use_bias=False)(input3)
    block5 = BatchNormalization(axis=axis)(block5)
    block5 = DepthwiseConv2D(depth_filters, use_bias=False,
                             depth_multiplier=D3,
                             depthwise_constraint=max_norm(1.))(block5)
    block5 = BatchNormalization(axis=axis)(block5)
    block5 = Activation('elu')(block5)
    block5 = AveragePooling2D(pool_size)(block5)
    block5 = dropoutType(dropoutRate)(block5)

    block6 = SeparableConv2D(F2_3, separable_filters3,
                             use_bias=False, padding='same')(block5)  # 36
    block6 = BatchNormalization(axis=axis)(block6)
    block6 = Activation('elu')(block6)
    block6 = AveragePooling2D(pool_size2)(block6)
    block6 = dropoutType(dropoutRate)(block6)
    block6 = Flatten()(block6)  # 41

    # 36 - 41

    merge_one = concatenate([block2, block4])
    merge_two = concatenate([merge_one, block6])

    flatten = Flatten()(merge_two)

    dense = Dense(nb_classes, name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)

    softmax = Activation('softmax', name='softmax')(dense)

    classifier = Model(inputs=[input1, input2, input3], outputs=softmax)
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return classifier

## NOISE ##

import numpy as np
import os
import mne
from scipy.io import loadmat

def add_awgn_noise(signal, snr):
    """
    Add AWGN noise to the signal based on a desired SNR.
    Args:
        signal (numpy.array): Original EEG signals.
        snr (float): Desired signal-to-noise ratio in dB.
    Returns:
        numpy.array: Signal with added Gaussian noise.
    """
    sig_power = np.mean(np.power(signal, 2))
    sig_db = 10 * np.log10(sig_power)
    noise_db = sig_db - snr
    noise_power = np.power(10, noise_db / 10)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    noisy_signal = signal + noise
    return noisy_signal
def create_noisy_dataset(data, snr):
    """
    Apply AWGN to each sample in the dataset.
    Args:
        data (numpy.array): Original EEG dataset.
        snr (float): Desired signal-to-noise ratio.
    Returns:
        numpy.array: Noisy EEG dataset.
    """
    noisy_data = np.array([add_awgn_noise(sample, snr) for sample in data])
    return noisy_data

from numpy import dot
from numpy.linalg import norm

def cosine_similarity(arr1, arr2):
    return dot(arr1.flatten(), arr2.flatten()) / (norm(arr1) * norm(arr2))

## RESULTS ##

def print_results(results):
    results_df = pd.DataFrame(results)

    avg = {
        'Accuracy': [np.mean(results_df['Accuracy'])],
        'F1': [np.mean(results_df['F1'])],
        'Precision': [np.mean(results_df['Precision'])],
        'Recall': [np.mean(results_df['Recall'])]
    }

    Avg = pd.DataFrame(avg)
    res_df = pd.concat([results_df, Avg])
    best_f1_index = results_df['F1'].idxmax()
    best_metrics = results_df.loc[best_f1_index, ['Accuracy', 'F1', 'Precision', 'Recall']]

    res_df.loc['Best'] = best_metrics

    index_vals = [f"F{i+1}" for i in range(len(res_df)-3)] + ['Test', 'Avg', 'Best']
    res_df.index = index_vals
    res_df.index.rename('Fold', inplace=True)
    print(res_df)
#############
## Archive ##
#############

## Models on full data


# def build_data_lstm_classifier(num_layers = 1):
#     classifier = Sequential()
#     #First Layer
#     classifier.add(ConvLSTM1D(filters=5, kernel_size=5, activation='relu', kernel_initializer='uniform', input_shape=(751, 22,1)))
#     classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu',  
#                          kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
#     classifier.add(Dropout(0.5))
#     # Intermediate Layers
#     for itr in range(num_layers):
#         classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', 
#                              kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
#         classifier.add(Dropout(0.5))   
#     # Last Layer
#     classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
#     classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
#     return classifier
# from keras.models import Sequential
# from keras.layers import ConvLSTM2D, Flatten, Dense, Dropout, Reshape
# from keras import regularizers


# def build_convlstm_classifier(num_layers=1):
#     classifier = Sequential()
    
#     # First ConvLSTM1D Layer
#     classifier.add(Reshape((-1, 64, 1), input_shape=( 751,22, 1)))
#     # classifier.add(ConvLSTM2D(32, kernel_size=(3, 1), activation='relu', input_shape=(None, 64)))
    
#     # Intermediate ConvLSTM1D Layers
#     for _ in range(num_layers):
#         classifier.add(ConvLSTM1D(64, kernel_size=3, activation='relu'))
    
#     # Flattening Layer
#     classifier.add(Flatten())
    
#     # Fully Connected Layers
#     classifier.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     classifier.add(Dropout(0.5))
    
#     # Output Layer
#     classifier.add(Dense(units=4, activation='softmax'))

#     # Compiling the model
#     classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return classifier


# def build_data_cnn_classifier(num_layers=1):
#     classifier = Sequential()
    
#     # First Convolutional Layer
#     classifier.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=(751, 22)))
#     classifier.add(MaxPooling1D(pool_size=2))
    
#     # Intermediate Convolutional Layers
#     for _ in range(num_layers):
#         classifier.add(Conv1D(64, kernel_size=3, activation='relu'))
#         classifier.add(MaxPooling1D(pool_size=2))
    
#     # Flattening Layer
#     classifier.add(Flatten())
    
#     # Fully Connected Layers
#     classifier.add(Dense(units=128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
#     classifier.add(Dropout(0.5))
    
#     # Output Layer
#     classifier.add(Dense(units=4, activation='softmax'))

#     # Compiling the model
#     classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    
#     return classifier


## CV Training

# acc = []
# ka = []
# prec = []
# recall = []

# for train_idx, test_idx in cv.split(labels):
    
#     Csp = [];ss = [];nn = [] # empty lists
    
#     label_train, label_test = labels[train_idx], labels[test_idx]
#     y_train, y_test = X_out[train_idx], X_out[test_idx]
    
#     # CSP filter applied separately for all Frequency band coefficients
    
#     Csp = [CSP(n_components=8, reg=None, log=True, norm_trace=False) for _ in range(8)]
#     ss = preprocessing.StandardScaler()

#     X_train = ss.fit_transform(np.concatenate(tuple(Csp[x].fit_transform(wpd_data[x,train_idx,:,:],label_train) for x  in range(8)),axis=-1))

#     X_test = ss.transform(np.concatenate(tuple(Csp[x].transform(wpd_data[x,test_idx,:,:]) for x  in range(8)),axis=-1))
#     nn = build_MLP()  
    
#     nn.fit(X_train, y_train, batch_size = 32, epochs = 5)
    
#     y_pred = nn.predict(X_test)
#     pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

#     acc.append(accuracy_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
#     ka.append(cohen_kappa_score(y_test.argmax(axis=1), pred.argmax(axis=1)))
#     prec.append(precision_score(y_test.argmax(axis=1), pred.argmax(axis=1),average='weighted'))
#     recall.append(recall_score(y_test.argmax(axis=1), pred.argmax(axis=1),average='weighted'))


# nn = build_cnn_classifier()  

# nn.fit(np.expand_dims(X_train,2), y_train, batch_size = 32, epochs = 25)

# y_pred = nn.predict(np.expand_dims(X_train,2))
# pred = (y_pred == y_pred.max(axis=1)[:,None]).astype(int)

## Results

# scores = {'Accuracy':acc,'Kappa':ka,'Precision':prec,'Recall':recall}

# Es = pd.DataFrame(scores)

# avg = {'Accuracy':[np.mean(acc)],'Kappa':[np.mean(ka)],'Precision':[np.mean(prec)],'Recall':[np.mean(recall)]}

# Avg = pd.DataFrame(avg)


# T = pd.concat([Es,Avg])

# T.index = ['F1','F2','F3','F4','F5','F6','F7','F8','F9','F10','Avg']
# T.index.rename('Fold',inplace=True)

# print(T)