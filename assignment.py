from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
tf.experimental.numpy.experimental_enable_numpy_behavior()
import numpy as np
import random


def linear_unit(x, W, b):
  return tf.matmul(x, W) + b

class ModelPart0:
    def __init__(self):
        """
        This model class contains a single layer network similar to Assignment 1.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        input = 32 * 32 * 3
        output = 2
        self.W1 = tf.Variable(tf.random.truncated_normal([input, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, output],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")


        self.trainable_variables = [self.W1, self.B1]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		# this reshape "flattens" the image data
        inputs = tf.reshape(inputs, [inputs.shape[0],-1])
        x = linear_unit(inputs, self.W1, self.B1)
        return x


class ModelPart1:
    def __init__(self):
        """
        This model class contains a multi-layer network similar to ModelPart0.
        """

        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # layer sizes
        input_size = 32 * 32 * 3 # flattened image size
        hidden_size = 256 # number of neurons in the hidden layer
        output_size = 2 # number of neurons in the output layer, number of classes, 2 for cat and dog

		# first layer
        self.W1 = tf.Variable(tf.random.truncated_normal([input_size, hidden_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, hidden_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

		# second layer
        self.W2 = tf.Variable(tf.random.truncated_normal([hidden_size, output_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, output_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")

		# list of trainable variables
        self.trainable_variables = [self.W1, self.B1, self.W2, self.B2]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)

		# this reshape "flattens" the image data
        inputs = tf.reshape(inputs, [inputs.shape[0],-1])

		# first layer
        hidden_layer = linear_unit(inputs, self.W1, self.B1)
		# apply ReLU activation function
        hidden_layer = tf.nn.relu(hidden_layer)

		# second layer
        logits = linear_unit(hidden_layer, self.W2, self.B2)

        return logits


class ModelPart3:
    def __init__(self):
        """
        A CNN with one convolutional layer followed by two fully connected layers.
        """
        # hyperparameters
        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # convolutional layer
        # filter parameters
        filter_height = 3
        filter_width = 3
        num_filters = 32 # num of feature detectors
        input_channels = 3 # RGB
        
        # init filters, [height, width, input_channels, in_channels, out_channels]
        self.conv_filters = tf.Variable(
            tf.random.truncated_normal([filter_height, filter_width, input_channels, num_filters],
                                    dtype=tf.float32, stddev=0.1),
            name="conv_filters"
        )

        # bias for each filter
        self.conv_biases = tf.Variable(
            tf.random.truncated_normal([num_filters],
                                    dtype=tf.float32, stddev=0.1),
            name="conv_biases"
        )
        
        # linear layers
        # after conv layer, image size is 32x32x32 = 32768 (with "SAME" padding)
        
        input_size = 32 * 32 * num_filters
        hidden_size = 256 # number of neurons in the hidden layer, same as ModelPart1
        output_size = 2 # number of neurons in the output layer, number of classes, 2 for cat and dog, binary classification

		# first layer
        self.W1 = tf.Variable(tf.random.truncated_normal([input_size, hidden_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W1")
        self.B1 = tf.Variable(tf.random.truncated_normal([1, hidden_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B1")

		# second layer
        self.W2 = tf.Variable(tf.random.truncated_normal([hidden_size, output_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="W2")
        self.B2 = tf.Variable(tf.random.truncated_normal([1, output_size],
                                                         dtype=tf.float32,
                                                         stddev=0.1),
                              name="B2")

		# list of trainable variables
        self.trainable_variables = [self.conv_filters, self.conv_biases, self.W1, self.B1, self.W2, self.B2]

    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        Forward pass: Convolution -> ReLU -> Flatten -> Dense -> ReLU -> Dense.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        
        # conv layer
        conv_output = tf.nn.conv2d(
            inputs, self.conv_filters, strides=[1, 1, 1, 1], padding="SAME"
        )
        conv_output = tf.nn.bias_add(conv_output, self.conv_biases) # add bias for broadcasting
        conv_output = tf.nn.relu(conv_output) # Non-linearity
        
        # this reshape "flattens" the image data, [batch_size, 32, 32, 32] -> [batch_size, 32768]
        conv_output_flat = tf.reshape(conv_output, [-1, 32 * 32 * 32])

        # first dense layer
        hidden_layer = linear_unit(conv_output_flat, self.W1, self.B1)
        # apply ReLU activation function
        hidden_layer = tf.nn.relu(hidden_layer)

        # second dense layer
        logits = linear_unit(hidden_layer, self.W2, self.B2)

        return logits


class ModelPart3Optional:
    def __init__(self):
        """
        A CNN with two conv layers, two max pooling layers, and two dense layers.
        """
        # Hyperparameters
        self.batch_size = 64
        self.num_classes = 2
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        # Conv Layer 1: 32 filters of 3x3, input has 3 channels (RGB)
        self.conv1_filters = tf.Variable(
            tf.random.truncated_normal([3, 3, 3, 32], dtype=tf.float32, stddev=0.1),
            name="conv1_filters"
        )
        self.conv1_biases = tf.Variable(
            tf.random.truncated_normal([32], dtype=tf.float32, stddev=0.1),
            name="conv1_biases"
        )

        # Conv Layer 2: 64 filters of 3x3, input has 32 channels from previous layer
        self.conv2_filters = tf.Variable(
            tf.random.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=0.1),
            name="conv2_filters"
        )
        self.conv2_biases = tf.Variable(
            tf.random.truncated_normal([64], dtype=tf.float32, stddev=0.1),
            name="conv2_biases"
        )

        # Dense Layers: After pooling, size is 8x8x64 = 4096
        flatten_size = 8 * 8 * 64
        hidden_size = 256
        self.W1 = tf.Variable(
            tf.random.truncated_normal([flatten_size, hidden_size], dtype=tf.float32, stddev=0.1),
            name="W1"
        )
        self.B1 = tf.Variable(
            tf.random.truncated_normal([1, hidden_size], dtype=tf.float32, stddev=0.1),
            name="B1"
        )
        self.W2 = tf.Variable(
            tf.random.truncated_normal([hidden_size, 2], dtype=tf.float32, stddev=0.1),
            name="W2"
        )
        self.B2 = tf.Variable(
            tf.random.truncated_normal([1, 2], dtype=tf.float32, stddev=0.1),
            name="B2"
        )

        self.trainable_variables = [
            self.conv1_filters, self.conv1_biases,
            self.conv2_filters, self.conv2_biases,
            self.W1, self.B1, self.W2, self.B2
        ]

    def call(self, inputs):
        """
        Forward pass: Conv -> Pool -> Conv -> Pool -> Flatten -> Dense -> Dense.
        :param inputs: shape (batch_size, 32, 32, 3)
        :return: logits, shape (batch_size, 2)
        """
        # Conv1: Scan for basic features
        conv1 = tf.nn.conv2d(inputs, self.conv1_filters, strides=[1, 1, 1, 1], padding="SAME")
        conv1 = tf.nn.bias_add(conv1, self.conv1_biases)
        conv1 = tf.nn.relu(conv1)

        # Pool1: Shrink by taking max in 2x2 areas
        pool1 = tf.nn.max_pool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Conv2: Scan for more complex features
        conv2 = tf.nn.conv2d(pool1, self.conv2_filters, strides=[1, 1, 1, 1], padding="SAME")
        conv2 = tf.nn.bias_add(conv2, self.conv2_biases)
        conv2 = tf.nn.relu(conv2)

        # Pool2: Shrink again
        pool2 = tf.nn.max_pool2d(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

        # Flatten: Turn 8x8x64 into a long vector
        flatten = tf.reshape(pool2, [-1, 8 * 8 * 64])

        # Dense1: Combine features
        hidden = linear_unit(flatten, self.W1, self.B1)
        hidden = tf.nn.relu(hidden)

        # Dense2: Make the final call
        logits = linear_unit(hidden, self.W2, self.B2)
        return logits
    

#--------------------------------
# Training
#--------------------------------
def loss(logits, labels):
	"""
	Calculates the cross-entropy loss after one forward pass.
	:param logits: during training, a matrix of shape (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	Softmax is applied in this function.
	:param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
	:return: the loss of the model as a Tensor
	"""

	# cross entropy loss softmax
	cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
	
	# average loss across batch
	mean_loss = tf.reduce_mean(cross_entropy)
	
	# add .numpy() to make assignment_tests.py loss test pass, otherwise, remove as breaks actual code.
	return mean_loss

def accuracy(logits, labels):
	"""
	Calculates the model's prediction accuracy by comparing
	logits to correct labels – no need to modify this.
	:param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
	containing the result of multiple convolution and feed forward layers
	:param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

	NOTE: DO NOT EDIT

	:return: the accuracy of the model as a Tensor
	"""
	correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

def train(model, train_inputs, train_labels):
	'''
	Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs
	and labels - ensure that they are shuffled in the same order using tf.gather.
	You should batch your inputs.
	:param model: the initialized model to use for the forward pass and backward pass
	:param train_inputs: train inputs (all inputs to use for training),
	shape (num_inputs, width, height, num_channels)
	:param train_labels: train labels (all labels to use for training),
	shape (num_labels, num_classes)
	:return: None
	'''
	# shuffle data (could be considered as part of preprocessing)
	# get indices
	indices = tf.range(len(train_inputs))
	# reorder them randomly
	shuffled_indices = tf.random.shuffle(indices)

	# get shuffled inputs
	shuffled_inputs = tf.gather(train_inputs, shuffled_indices)
	# get shuffled labels
	shuffled_labels = tf.gather(train_labels, shuffled_indices)

	# find number of batches (number of inputs / batch size)
	num_batches = len(train_inputs) // model.batch_size

	# process through the batches
	for i in range(num_batches):
		# get batch indices
		start_idx = i * model.batch_size
		end_idx = start_idx + model.batch_size

		# get batch inputs
		batch_inputs = shuffled_inputs[start_idx:end_idx]
		# get batch labels
		batch_labels = shuffled_labels[start_idx:end_idx]
            
		# reshape inputs before model.call()
		batch_inputs = tf.reshape(batch_inputs, [model.batch_size, 32, 32, 3])

		# forward pass and backwards pass with gradient tape
		with tf.GradientTape() as tape:
			# get predictions
			logits = model.call(batch_inputs)
			# compute loss
			batch_loss = loss(logits, batch_labels)

		# compute gradients, how to adjust weights and bias
		gradients = tape.gradient(batch_loss, model.trainable_variables)

		# update weights
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
		
		# Print batch loss after updating weights
		# print(batch_loss)


def test(model, test_inputs, test_labels):
	"""
	Tests the model on the test inputs and labels.
	:param test_inputs: test data (all images to be tested),
	shape (num_inputs, width, height, num_channels)
	:param test_labels: test labels (all corresponding labels),
	shape (num_labels, num_classes)
	:return: test accuracy - this can be the average accuracy across
	all batches or the sum as long as you eventually divide it by batch_size
	"""
	return accuracy(model.call(test_inputs), test_labels)


def visualize_results(image_inputs, probabilities, image_labels, first_label, second_label):
	"""
	Uses Matplotlib to visualize the results of our model.
	:param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
	:param probabilities: the output of model.call(), shape (10, num_classes)
	:param image_labels: the labels from get_data(), shape (10, num_classes)
	:param first_label: the name of the first class, "dog"
	:param second_label: the name of the second class, "cat"

	NOTE: DO NOT EDIT

	:return: doesn't return anything, a plot should pop-up
	"""
	predicted_labels = np.argmax(probabilities, axis=1)
	num_images = image_inputs.shape[0]

	fig, axs = plt.subplots(ncols=num_images)
	fig.suptitle("PL = Predicted Label\nAL = Actual Label")
	for ind, ax in enumerate(axs):
			ax.imshow(image_inputs[ind], cmap="Greys")
			pl = first_label if predicted_labels[ind] == 0.0 else second_label
			al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
			ax.set(title="PL: {}\nAL: {}".format(pl, al))
			plt.setp(ax.get_xticklabels(), visible=False)
			plt.setp(ax.get_yticklabels(), visible=False)
			ax.tick_params(axis='both', which='both', length=0)
	plt.show()


CLASS_CAT = 3
CLASS_DOG = 5
def main(cifar10_data_folder):
    '''
    Read in CIFAR10 data (limited to 2 classes), initialize your model, and train and
    test your model for a number of epochs. We recommend that you train for
    25 epochs.
    You should receive a final accuracy on the testing examples for cat and dog
    of ~60% for Part1 and ~70% for Part3.
    :return: None
    '''
    # Load the training and testing data
    train_inputs, train_labels = get_data(os.path.join(cifar10_data_folder, 'train'), CLASS_CAT, CLASS_DOG)
    test_inputs, test_labels = get_data(os.path.join(cifar10_data_folder, 'test'), CLASS_CAT, CLASS_DOG)

    # Initialize the model
    # model = ModelPart0()
    # model = ModelPart1()
    # model = ModelPart3()
    # model = ModelPart3Optional()

    # Train the model for 25 epochs
    num_epochs = 25
    # for epoch in range(num_epochs):
    # 	print(f"Epoch {epoch+1}/{num_epochs}")
    # 	train(model, train_inputs, train_labels)

    # Test the model and print accuracy
    # test_accuracy = test(model, test_inputs, test_labels)
    # print(f"Test accuracy: {test_accuracy}")
      
    # Test ModelPart0 (58-60% accuracy)
    model = ModelPart0()
    for epoch in range(num_epochs):
        train(model, train_inputs, train_labels)
        print(f"ModelPart0 Epoch {epoch+1}/{num_epochs}")
    test_accuracy = test(model, test_inputs, test_labels)
    print(f"ModelPart0 accuracy: {test_accuracy}")

    # Optional: Visualize predictions on 10 test images
    vis_inputs = test_inputs[:10]
    vis_labels = test_labels[:10]
    logits = model.call(vis_inputs)
    probabilities = tf.nn.softmax(logits)  # Convert logits to probabilities
    visualize_results(vis_inputs, probabilities, vis_labels, "cat", "dog")

    # Test ModelPart1 (60-63% accuracy)
    model = ModelPart1()
    for epoch in range(num_epochs):
        train(model, train_inputs, train_labels)
        print(f"ModelPart1 Epoch {epoch+1}/{num_epochs}")
    test_accuracy = test(model, test_inputs, test_labels)
    print(f"ModelPart1 accuracy: {test_accuracy}")

    # Optional: Visualize predictions on 10 test images
    vis_inputs = test_inputs[:10]
    vis_labels = test_labels[:10]
    logits = model.call(vis_inputs)
    probabilities = tf.nn.softmax(logits)  # Convert logits to probabilities
    visualize_results(vis_inputs, probabilities, vis_labels, "cat", "dog")

    # Test ModelPart3 (70% accuracy)
    model = ModelPart3()
    for epoch in range(num_epochs):
        train(model, train_inputs, train_labels)
        print(f"ModelPart3 Epoch {epoch+1}/{num_epochs}")
    test_accuracy = test(model, test_inputs, test_labels)
    print(f"ModelPart3 accuracy: {test_accuracy}")

    # Optional: Visualize predictions on 10 test images
    vis_inputs = test_inputs[:10]
    vis_labels = test_labels[:10]
    logits = model.call(vis_inputs)
    probabilities = tf.nn.softmax(logits)  # Convert logits to probabilities
    visualize_results(vis_inputs, probabilities, vis_labels, "cat", "dog")
    
    # Test ModelPart3Optional (70-75% accuracy)
    model = ModelPart3Optional()
    for epoch in range(num_epochs):
        train(model, train_inputs, train_labels)
        print(f"ModelPart3Optional Epoch {epoch+1}/{num_epochs}")
    test_accuracy = test(model, test_inputs, test_labels)
    print(f"ModelPart3Optional accuracy: {test_accuracy}")

    # Optional: Visualize predictions on 10 test images
    vis_inputs = test_inputs[:10]
    vis_labels = test_labels[:10]
    logits = model.call(vis_inputs)
    probabilities = tf.nn.softmax(logits)  # Convert logits to probabilities
    visualize_results(vis_inputs, probabilities, vis_labels, "cat", "dog")


if __name__ == '__main__':
    # default = './CIFAR_data/'
    cifar_data_folder = './CIFAR_data/'
    main(cifar_data_folder)
