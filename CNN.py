'''
Convolutional Neural Network Layer
Implementation of the Convolutional Layer as part of the CLDNN Architecture

Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
Adopted and Modified From: http://deeplearning.net/tutorial/lenet.html
'''
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv2d
import numpy as np

class CNNLayer(object):
	def __init__(self, input, filter_shape, image_shape, poolsize):
		self.rng = np.random.RandomStates()		# Initialize RNG
	
		assert filter_shape == image_shape		# Assert Filter and Image Dimensions
		self.input = input						# Localize Input to CNNLayer Object

		# Initialize Weight Bounds - PDF on [-1/fan_in, 1/fan_out]
        fan_in = np.prod(filter_shape[1:])
		fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) np.prod(poolsize))
		W_bound = np.sqrt(6. / (fan_in + fan_out))	# Where did this 6 come from?

		# Initialize Weights Randomly Based on Bounds
		self.W = theano.shared(
			np.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
			),
			borrow=True
        )

		# Initialize 1D Bias Tensor for Each Feature Map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)
		
		# Initialize Input Feature Map with Filters
        conv_out = conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            image_shape = image_shape
        )

		

		
