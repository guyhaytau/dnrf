import tensorflow as tf
from tensorflow import keras
import numpy as np
from routing_tree import rft3, LEFT, RIGHT

DEFAULT_NUMBER_OF_LAYERS = 5
DEFAULT_NUMBER_OF_TREES = 10
BATCH_SIZE = 258
CNN_W = "cnn_w"
R_W = "r_w"
FINAL_PROBABILITIES = "final_probabilities"

class DifferentiableRandomForestLayer(keras.layers.Layer):
	"""docstring for DifferentiableRandomForestLayer"""
	def __init__(self, n_classes, n_trees = DEFAULT_NUMBER_OF_TREES,
				n_layers = DEFAULT_NUMBER_OF_LAYERS,
				final_probabilities = None):
		super(DifferentiableRandomForestLayer, self).__init__()
		# Number of layers for each tree
		self.n_layers = n_layers
		# Number of end nodes for each tree
		self.n_end_nodes =  2 ** (self.n_layers - 1)
		# Number of leafs
		self.n_leafs = 2 * self.n_end_nodes
		# Number of nodes for each tree
		self.n_nodes = np.sum(2 ** np.arange(self.n_layers))
		self.n_classes = n_classes
		self.n_trees = DEFAULT_NUMBER_OF_TREES
		# Number of leafs that are directly connected to the CNN
		self.connection_leafs = self.n_nodes
		self.tree_trained = 0


	def build(self, input_shape):
		# Creating weights from the cnn layer to the trees
		self.cnn_w = []
		self.final_probabilities = []
		# for each tree the probabitilty of reaching the specific leaf
		self.routing = [i for i in range(self.n_trees)]

		for i in range(self.n_trees):
			# Cnn weights initializer
			cnn_initializer = tf.keras.initializers.RandomNormal()
			# Initializing weight from cnn to tree nodes
			self.cnn_w.append(self.add_weight(CNN_W + str(i),
								shape=[int(input_shape[-1]), self.connection_leafs],
								initializer = cnn_initializer,
								trainable = True))

			# Initializing final probabities uniformaly
			self.final_probabilities.append(self.add_weight(FINAL_PROBABILITIES + str(i),
											shape=[self.n_leafs, self.n_classes],
											initializer = keras.initializers.Constant(value = self.n_classes ** -1),
											trainable = True))

	def call(self, l_input, training):
		tree_indexes_to_update = range(self.n_trees)
		n_t = np.random.randint(0,self.n_trees)

		for n_t in tree_indexes_to_update:
			cnn_w = self.cnn_w[n_t]
			# Probability dictating if to go left
			d_tree = tf.nn.sigmoid(tf.matmul(l_input, cnn_w))
			# If to go right
			d_tree_hat = tf.subtract(tf.ones_like(d_tree), d_tree)
			# Splits the final probs and then multiplies by left or right
			# Of the current node probs
			split_size = self.final_probabilities[n_t].shape[0]
			tot_prob = tf.ones(self.n_leafs)
			node_counter = 0
			for l in range(self.n_layers):
				split_size = int(split_size / 2)
				prob = []
				# Number of iteration of loop also equals number of nodes in layer
				iteration_num = int((self.final_probabilities[n_t].shape[0] / split_size) / 2)
				for i in range(iteration_num):
					left = tf.ones(split_size) * tf.gather(d_tree, [node_counter], axis = 1)
					right = tf.ones(split_size) * tf.gather(d_tree_hat, [node_counter], axis = 1)
					prob.append(tf.concat([left,right], axis = 1))
					# Continuing on to next node
					node_counter += 1
				tot_prob = tf.math.multiply(tf.concat(prob, axis = 1), tot_prob)
			# BATCH X self.n_leafs
			self.routing[n_t] = tot_prob

		if training:
			self.tree_trained = np.random.randint(0, self.n_trees)
			prob = tf.matmul(self.routing[self.tree_trained], self.final_probabilities[self.tree_trained])
		else:
			prob = tf.matmul(self.routing[0], self.final_probabilities[0])
			for i in range(1,self.n_trees):
				prob += tf.matmul(self.routing[i], self.final_probabilities[i])
			prob = tf.math.divide_no_nan(prob, self.n_trees)
		# When algorithum becomes to sure, sometimes due to rounding error
		# get 1.0000000001 prodiction value and crash
		prob = tf.clip_by_value(prob, 0, 1)
		return prob