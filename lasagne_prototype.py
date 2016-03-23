'''
CLDNN Architecture Prototype Implementation in Lasagne
Simple optimizations made to test out the architecture. Later can be 
implemented separately by actual Theano source to optimize training.

Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import numpy as np
import theano
import theano.tensor as T

import lasagne
from lasagne.layers import *

#input_var = T.tensor4('X')
#target_var = T.ivector('y')

def buildDataset(location):
	print 'Building Dataset...'

def buildCLDNN(input_var = None):
	# Input Layer - TODO: Compute the size of the input based on image size
	input_layer = InputLayer( (None, 1, width, height), input_var )

	# Convolutional Layer (15 x 15) 
