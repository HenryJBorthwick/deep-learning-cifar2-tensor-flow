from __future__ import absolute_import

import os
import tensorflow as tf
import numpy as np
import random
import math

def conv2d(inputs, filters, strides, padding):
	"""
	Performs 2D convolution given 4D inputs and filter Tensors.
	:param inputs: tensor with shape [num_examples, in_height, in_width, in_channels]
	:param filters: tensor with shape [filter_height, filter_width, in_channels, out_channels]
	:param strides: MUST BE [1, 1, 1, 1] - list of strides, with each stride corresponding to each dimension in input
	:param padding: either "SAME" or "VALID", capitalization matters
	:return: outputs, NumPy array or Tensor with shape [num_examples, output_height, output_width, output_channels]
	"""
	num_examples = None
	in_height = None
	in_width = None
	input_in_channels = None

	filter_height = None
	filter_width = None
	filter_in_channels = None
	filter_out_channels = None

	num_examples_stride = None
	strideY = None
	strideX = None
	channels_stride = None

	# ------------------------------------------------------------
	# Extract Dimensions	
	# ------------------------------------------------------------

	# get dimensions from input
	num_examples, in_height, in_width, input_in_channels = inputs.shape
	# get dimensions from filters
	filter_height, filter_width, filter_in_channels, filter_out_channels = filters.shape

	# number of input channels must match filers input channels
	assert input_in_channels == filter_in_channels, "Input and filter must have the same number of input channels."

	# strides must be [1, 1, 1, 1], moves filter 1 pixel at a time
	num_examples_stride, strideY, strideX, channels_stride = strides
	assert strides == [1, 1, 1, 1], "Strides must be [1, 1, 1, 1] as specified."
		
	# ------------------------------------------------------------
	# Padding and Output Dimensions
	# ------------------------------------------------------------

	# Cleaning padding input
	# padding is for when the filter is at the edge of the input
	# calculate how much it hangs off the edge with: (filter_size - 1) / 2
	# 5x5 image, 3x3 filter, output is 5 - 3 + 1 = 3, so 3x3
	if padding == "SAME":
		padY = (filter_height - 1) // 2  # Integer division for top/bottom
		padX = (filter_width - 1) // 2   # Left/right
		output_height = in_height
		output_width = in_width
	elif padding == "VALID":
		padY = 0
		padX = 0
		output_height = in_height - filter_height + 1
		output_width = in_width - filter_width + 1
	else:
		raise ValueError("Padding must be 'SAME' or 'VALID'.")

	# pad input with zeros since inputs is 4D, we pad width and height
	padded_inputs = np.pad(inputs, [(0, 0), (padY, padY), (padX, padX), (0, 0)], mode='constant')

	#--------------------------------
	# Convolution
	#--------------------------------
	# Calculate output dimensions

	# init output
	outputs = np.zeros((num_examples, output_height, output_width, filter_out_channels))

	# convolve loop
	for n in range(num_examples): # image
		for k in range(filter_out_channels): # filter
			for i in range(output_height): # output height (i)
				for j in range(output_width): # output width (j)
					# get the patch from padded_inputs
					patch = padded_inputs[n, i:i+filter_height, j:j+filter_width, :]
					# get the k-th filter
					filter_k = filters[:, :, :, k]
					# multiply and sum
					outputs[n, i, j, k] = np.sum(patch * filter_k)

	return outputs
