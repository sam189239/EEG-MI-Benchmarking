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

def build_mlp_classifier(num_layers = 1):
    classifier = Sequential()
    classifier.add(Flatten())
    #First Layer
    classifier.add(Dense(units = 256, kernel_initializer = 'uniform', activation = 'relu', 
                         kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
    classifier.add(Dropout(0.5))
    # Intermediate Layers
    for itr in range(num_layers):
        classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', 
                             kernel_regularizer=regularizers.l2(0.01))) # L2 regularization
        classifier.add(Dropout(0.5))   
    # Last Layer
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'softmax'))
    classifier.compile(optimizer = 'rmsprop' , loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return classifier


def build_cnn_classifier(input_shape, num_layers=1):
    classifier = Sequential()
    
    # First Convolutional Layer
    classifier.add(Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    classifier.add(Conv1D(32, kernel_size=3, activation='relu'))
    classifier.add(MaxPooling1D(pool_size=2))
    
    # Intermediate Convolutional Layers
    for _ in range(num_layers):
        classifier.add(Conv1D(64, kernel_size=3, activation='relu'))
        classifier.add(MaxPooling1D(pool_size=2))
    
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