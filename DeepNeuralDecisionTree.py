import tensorflow as tf
from tensorflow import keras
from os import path
from DifferentiableRandomForestLayer import CNN_W,R_W,FINAL_PROBABILITIES
import numpy as np

WEIGHT = 'weight'
GRAD = 'grad'

class DeepNeuralDecisionTree(keras.Sequential):
	def __init__(self, *args, **kargs):
		super(DeepNeuralDecisionTree, self).__init__(*args, **kargs)
		self.node_parameter_future_change = []
		self.ep = 0

	def on_epoch_end(self, *args, **kargs):
		weights = self.get_weights()
		flayer, fweights_dict = self.get_forest_layer_and_weight_dict()
		for i, change in enumerate(self.node_parameter_future_change):
			normalizing_factor = tf.reshape(tf.reduce_sum(change, axis = 1), [change.shape[0], 1])
			self.node_parameter_future_change[i] = tf.math.divide_no_nan(change, normalizing_factor)
			weight_index = fweights_dict[FINAL_PROBABILITIES][i][GRAD]
			weights[weight_index] = self.node_parameter_future_change[i]
			# Printing node parameters after last epoch, for analysis
			# should be cleaner
			if args[0] == 99:
				print("*"*60)
				print("Final Node Parameters For: {0}".format(i))
				print(self.node_parameter_future_change[i])
		self.set_weights(weights)

	def train_step(self, data):
		inputs, targets = data
		trainable_vars = self.trainable_variables
		with tf.GradientTape() as tape1:
			preds = self(inputs, training=True)  # Forward pass
			loss = self.compiled_loss(targets, preds)
		# Compute first-order gradients
		grads = tape1.gradient(loss, trainable_vars)

		flayer, fweights_dict = self.get_forest_layer_and_weight_dict()

		# Initializing node_parameter_future_change
		if len(self.node_parameter_future_change) == 0:
			self.node_parameter_future_change = [tf.zeros_like(fweights_dict[FINAL_PROBABILITIES][i][WEIGHT]) for i in range(flayer.n_trees)]

		# Updating final probabilities for one tree
		# Disabling grad change
		for i in range(flayer.n_trees):
			if i == flayer.tree_trained:
				self.update_final_prob_tree(i, fweights_dict, preds, targets, grads)
				grads[fweights_dict[FINAL_PROBABILITIES][i][GRAD]] = tf.zeros_like(fweights_dict[FINAL_PROBABILITIES][i][WEIGHT])
			else:
				grads[fweights_dict[FINAL_PROBABILITIES][i][GRAD]] = tf.zeros_like(fweights_dict[FINAL_PROBABILITIES][i][WEIGHT])

		# Update weights
		self.optimizer.apply_gradients(zip(grads, trainable_vars))

		# Update metrics (includes the metric that tracks the loss)
		self.compiled_metrics.update_state(targets, preds)

		# Return a dict mapping metric names to current value
		return {m.name: m.result() for m in self.metrics}


	def get_forest_layer_and_weight_dict(self):
		# Getting the forest layer and creating the forest layer weight dict

		forest_layer = None
		# Will hold the index in the weights, grads calculated
		grad_index = -1
		aa = self.get_weights()
		for l in self.layers:
			if l.name == 'differentiable_random_forest_layer':
				forest_layer = l
				break
			else:
				grad_index += len(l.trainable_variables)
		assert(forest_layer != None)
		weights_dict = {CNN_W : {}, R_W : {}, FINAL_PROBABILITIES: {}}
		names = [weight.name for weight in forest_layer.trainable_variables]
		trainable_weights = forest_layer.trainable_variables

		for name, weight in zip(names, trainable_weights):
			grad_index += 1
			name = path.split(name)[-1]
			if CNN_W in name:
				w_n = int(name.split(':')[0][len(CNN_W):])
				weights_dict[CNN_W][w_n] = {WEIGHT: weight, GRAD: grad_index}

			if FINAL_PROBABILITIES in name:
				w_n = int(name.split(':')[0][len(FINAL_PROBABILITIES):])
				weights_dict[FINAL_PROBABILITIES][w_n] = {WEIGHT: weight, GRAD: grad_index}

		for i, weight in enumerate(forest_layer.routing):
			name = R_W + str(i)
			w_n = int(name.split(':')[0][len(R_W):])
			weights_dict[R_W][w_n] = {WEIGHT: weight, GRAD: None}

		return forest_layer, weights_dict

	def update_final_prob_tree(self, tree_index, fweights_dict, preds, real_y, grads):
		# BATCH X self.n_leafs
		routing_w = fweights_dict[R_W][tree_index][WEIGHT]
		# self.n_leafs X 10
		final_w = fweights_dict[FINAL_PROBABILITIES][tree_index][WEIGHT]
		update_index = fweights_dict[FINAL_PROBABILITIES][tree_index][GRAD]
		final_probs = []
		for i in range(final_w.shape[0]):
			# Following algorithm to iterate final probabilities
			# BATCH X 1
			leaf_route = tf.gather(routing_w, [i], axis = 1)
			# 1 X 10
			leaf_prob = tf.gather(final_w, [i], axis = 0)
			# BATCH X 10
			final_leaf_y_prob = tf.matmul(leaf_route, leaf_prob)
			# Batch X 10
			right_prob = tf.multiply(final_leaf_y_prob, real_y)

			probs = tf.math.divide_no_nan(right_prob,preds)
			probs = tf.reduce_sum(probs, axis = 0)
			probs = tf.reshape(probs, [1,probs.shape[0]])
			final_probs.append(probs)
		final_probs = tf.concat(final_probs, axis = 0)
		self.node_parameter_future_change[tree_index] = tf.add(self.node_parameter_future_change[tree_index], final_probs)


		