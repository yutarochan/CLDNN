"""
Convolutional LSTM Deep Neural Network
Implementation of the Convolutional LSTM DNN Architecture for Spatio-Temporal Classification of Videos

Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)

References:
- 
"""
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

''' 
Load Dataset Into Training, Validation, Testing 
TODO: Develop dataset util loading specificially for PACO
'''
#datasets = load_data(dataset)
#train_set_x, train_set_y
#valid_set_x, valid_set_y
#test_set_x, test_set_y

'''
Phase 1: Convolution Layer + Pooling
Implementation as local filters for visual features
'''

'''
Phase 2: Linear Layer
Perform dimensionality reduction prior to LSTM Phase
'''

'''
Phase 3: LSTM Layer
'''

'''
Phase 4: DNN Fully Connected Layer
'''
