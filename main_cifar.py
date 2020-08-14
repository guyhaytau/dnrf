import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, mnist
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold
from DeepNeuralDecisionTree import DeepNeuralDecisionTree

import matplotlib.gridspec as gridspec
from DifferentiableRandomForestLayer import DifferentiableRandomForestLayer

# 1 - 0, will be used as the threshold for recall and precision matrices
THRESHOLD = 0.5
# plot diagnostic learning curves
def summarize_diagnostics(history, name):
	gs = gridspec.GridSpec(2,2)
	fig = plt.figure(tight_layout = True)
	ax1 = fig.add_subplot(gs[0,:])
	# plot loss
	# plt.subplot(211)
	ax1.set_title('Cross Entropy Loss')
	ax1.plot(history.history['loss'], color='blue', label='train')
	ax1.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	# plt.subplot(213)
	ax2 = fig.add_subplot(gs[1,:])
	ax2.set_title('Classification Accuracy')
	ax2.plot(history.history['categorical_accuracy'], color='blue', label='train')
	ax2.plot(history.history['val_categorical_accuracy'], color='orange', label='test')

	# save plot to file
	plt.savefig('{0}_plot.png'.format(name))
	plt.close()

# 3-block VGG
def create_model_ci_3():
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same', input_shape=(32,32,3)))
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(128, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.Conv2D(128, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation = tf.nn.relu, kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation = tf.nn.softmax))

	opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, 
					loss='categorical_crossentropy',
					metrics = [keras.metrics.CategoricalAccuracy(), 
								keras.metrics.Precision(thresholds = THRESHOLD), 
								keras.metrics.Recall(thresholds = THRESHOLD),
								keras.metrics.TopKCategoricalAccuracy(k=2)])
	return model

# 2-block VGG
def create_model_ci_2():
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same', input_shape=(32,32,3)))
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, activation = tf.nn.relu, kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation = tf.nn.softmax))

	opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, 
					loss='categorical_crossentropy',
					metrics = [keras.metrics.CategoricalAccuracy(), 
								keras.metrics.Precision(thresholds = THRESHOLD), 
								keras.metrics.Recall(thresholds = THRESHOLD),
								keras.metrics.TopKCategoricalAccuracy(k=2)])
	return model

# 1-block VGG
def create_model_ci_1():
	model = keras.Sequential()
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same', input_shape=(32,32,3)))
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(32, activation = tf.nn.relu, kernel_initializer='he_uniform'))
	model.add(keras.layers.Dense(10, activation = tf.nn.softmax))

	opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9)
	model.compile(optimizer = opt, 
					loss='categorical_crossentropy',
					metrics = [keras.metrics.CategoricalAccuracy(), 
								keras.metrics.Precision(thresholds = THRESHOLD), 
								keras.metrics.Recall(thresholds = THRESHOLD),
								keras.metrics.TopKCategoricalAccuracy(k=2)])
	return model

# 3-block VGG with Random Forest layer
def create_dndf_ci_3():
	model = DeepNeuralDecisionTree()
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same', input_shape=(32,32,3)))
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(128, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.Conv2D(128, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Dropout(0.25))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(128, activation = tf.nn.relu, kernel_initializer='he_uniform'))
	model.add(DifferentiableRandomForestLayer(10))

	opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = False)
	model.compile(optimizer = opt, 
					loss=keras.losses.CategoricalCrossentropy(),
					metrics = [keras.metrics.CategoricalAccuracy(), 
								keras.metrics.Precision(thresholds = THRESHOLD), 
								keras.metrics.Recall(thresholds = THRESHOLD),
								keras.metrics.TopKCategoricalAccuracy(k=2)],
					run_eagerly = True)

	return model

# 2-block VGG with Random Forest layer
def create_dndf_ci_2():
	model = DeepNeuralDecisionTree()
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same', input_shape=(32,32,3)))
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.Conv2D(64, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(64, activation = tf.nn.relu, kernel_initializer='he_uniform'))
	model.add(DifferentiableRandomForestLayer(10))

	opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = False)
	model.compile(optimizer = opt, 
					loss=keras.losses.CategoricalCrossentropy(),
					metrics = [keras.metrics.CategoricalAccuracy(), 
								keras.metrics.Precision(thresholds = THRESHOLD), 
								keras.metrics.Recall(thresholds = THRESHOLD),
								keras.metrics.TopKCategoricalAccuracy(k=2)],
					run_eagerly = True)

	return model

# 1-block VGG with Random Forest layer
def create_dndf_ci_1():
	model = DeepNeuralDecisionTree()
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same', input_shape=(32,32,3)))
	model.add(keras.layers.Conv2D(32, (3,3), activation = tf.nn.relu, kernel_initializer = 'he_uniform', padding='same'))
	model.add(keras.layers.MaxPooling2D((2,2)))
	model.add(keras.layers.Flatten())
	model.add(keras.layers.Dense(32, activation = tf.nn.relu, kernel_initializer='he_uniform'))
	model.add(DifferentiableRandomForestLayer(10))

	opt = keras.optimizers.SGD(lr = 0.001, momentum = 0.9, nesterov = False)
	model.compile(optimizer = opt, 
					loss=keras.losses.CategoricalCrossentropy(),
					metrics = [keras.metrics.CategoricalAccuracy(), 
								keras.metrics.Precision(thresholds = THRESHOLD), 
								keras.metrics.Recall(thresholds = THRESHOLD),
								keras.metrics.TopKCategoricalAccuracy(k=3)],
					run_eagerly = True)

	return model

def main():
	(train_x, train_y), (test_x, test_y) = cifar10.load_data()
	x = np.concatenate((train_x, test_x))
	y = np.concatenate((train_y, test_y))
	# normalizing to 0-1
	x = x.astype('float32')
	x /= 255
	skf = StratifiedKFold(n_splits = 5, shuffle = True)
	skf.get_n_splits(x,y)

	new_model = create_dndf_ci_3()
	accuracies = []
	precisions = []
	recalls = []
	top_k_accuracies = []

	# To avoid recompiling
	model_init_weights = new_model.get_weights()
	counter = 0
	for train_index, test_index in skf.split(x, y):
		counter += 1
		train_x, test_x = x[train_index], x[test_index]
		train_y, test_y= y[train_index], y[test_index]

		# Removing categorical bais changing from 0-10 to vector
		test_y = keras.utils.to_categorical(test_y)
		train_y = keras.utils.to_categorical(train_y)
		print("*"*60)
		identifing_string = "three_vg_cifar_10_trees_100_dropout_{0}".format(counter)

		# if counter <= 4:
		# 	new_model = keras.models.load_model(identifing_string)
		# else:
		history = new_model.fit(train_x, train_y, epochs = 100, batch_size = 100, validation_data = (test_x, test_y), callbacks = [keras.callbacks.LambdaCallback(on_epoch_end = new_model.on_epoch_end)])
		# history = new_model.fit(train_x, train_y, epochs = 100, batch_size = 1000, validation_data = (test_x, test_y))
		new_model.save(identifing_string)
		summarize_diagnostics(history, identifing_string)
		loss, acc, precision, recall, top_k_acc = new_model.evaluate(test_x, test_y)
		print("loss: {0}, acc: {1}, precision: {2}, recall: {3}, top k acc: {4}".format(loss, acc, precision, recall, top_k_acc))
		accuracies.append(acc)
		precisions.append(precision)
		recalls.append(recall)
		top_k_accuracies.append(top_k_acc)

		# Next loop won't be trained
		new_model.set_weights(model_init_weights)

	final_title = "I'm VG3 cifar 10 trees 100 batch size Dropout"
	print(final_title)
	print("Average Accuracy: {0} Standard Deviation: {1}".format(np.mean(accuracies), np.std(accuracies, ddof = 1)))
	print("Average Percision: {0} Standard Deviation: {1}".format(np.mean(precisions), np.std(precisions, ddof = 1)))
	print("Average Recall: {0} Standard Deviation: {1}".format(np.mean(recalls), np.std(recalls, ddof = 1)))
	print("Average Top K Accuracy: {0} Standard Deviation: {1}".format(np.mean(top_k_accuracies), np.std(top_k_accuracies, ddof = 1)))

	with open(identifing_string, 'w') as f:
		f.write(final_title)
		f.write("Average Accuracy: {0} Standard Deviation: {1}".format(np.mean(accuracies), np.std(accuracies, ddof = 1)))
		f.write("Average Percision: {0} Standard Deviation: {1}".format(np.mean(precisions), np.std(precisions, ddof = 1)))
		f.write("Average Recall: {0} Standard Deviation: {1}".format(np.mean(recalls), np.std(recalls, ddof = 1)))
		f.write("Average Top K Accuracy: {0} Standard Deviation: {1}".format(np.mean(top_k_accuracies), np.std(top_k_accuracies, ddof = 1)))


if __name__ == '__main__':
	main()