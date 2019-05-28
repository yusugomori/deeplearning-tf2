import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
	Conv2D, Dense, MaxPool2D, Flatten, Input
import numpy as np
from sklearn.utils import shuffle

class Vgg16(Model):
	def __init__(self,output_nodes):
		super(Vgg16, self).__init__()
		self.conv1_1 = Conv2D(
					input_shape=[None,28,28,1],filters=64, kernel_size=3,
					padding='same', activation='relu')
		self.conv1_2 = Conv2D(
					filters=64, kernel_size=3,
					padding='same', activation='relu')
		self.conv2_1 = Conv2D(
					filters=128, kernel_size=3,
					padding='same', activation='relu')
		self.conv2_2 = Conv2D(
					filters=128, kernel_size=3,
					padding='same', activation='relu')
		self.conv3_1 = Conv2D(
					filters=256,kernel_size=3,
					padding='same', activation='relu')
		self.conv3_2 = Conv2D(
					filters=256,kernel_size=3,
					padding='same', activation='relu')
		self.conv3_3 = Conv2D(
					filters=256,kernel_size=3,
					padding='same', activation='relu')
		
		self.conv4_1 = Conv2D(
					filters=512, kernel_size=3,
					padding='same', activation='relu')
		self.conv4_2 = Conv2D(
					filters=512, kernel_size=3,
					padding='same', activation='relu')
		self.conv4_3 = Conv2D(
					filters=512, kernel_size=3,
					padding='same', activation='relu')
		self.conv5_1 = Conv2D(
					filters=512, kernel_size=3,
					padding='same', activation='relu')
		self.conv5_2 = Conv2D(
					filters=512, kernel_size=3,
					padding='same', activation='relu')
		self.conv5_3 = Conv2D(
					filters=512, kernel_size=3,
					padding='same', activation='relu')
		self.dense1_1 = Dense(
					units=4096, activation='relu')
		self.dense1_2 = Dense(
					units=4096, activation='relu')
		self.dense2 = Dense(
					units=output_nodes, activation='softmax')
		self.maxPool = MaxPool2D(
					pool_size=2, strides=2, padding='same')
		self.flatten = Flatten()

	def call(self,input):
		x = self.conv1_1(input)
		x = self.conv1_2(x)
		x = self.maxPool(x)
		x = self.conv2_1(x)
		x = self.conv2_2(x)
		x = self.maxPool(x)
		x = self.conv3_1(x)
		x = self.conv3_2(x)
		x = self.conv3_3(x)
		x = self.maxPool(x)
		x = self.conv4_1(x)
		x = self.conv4_2(x)
		x = self.conv4_3(x)
		x = self.maxPool(x)
		x = self.conv5_1(x)
		x = self.conv5_2(x)
		x = self.conv5_3(x)
		x = self.maxPool(x)
		x = self.flatten(x)
		x = self.dense1_1(x)
		x = self.dense1_2(x)
		x = self.dense2(x)
		return x


if __name__ == '__main__':
	
	@tf.function
	def compute_loss(label, pred):
		return criterion(label, pred)

	@tf.function
	def train_step(x, t):
		with tf.GradientTape() as tape:
			preds = model(x)
			loss = compute_loss(t, preds)
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))
		train_loss(loss)
		train_acc(t, preds)

		return preds

	@tf.function
	def test_step(x, t):
		preds = model(x)
		loss = compute_loss(t, preds)
		test_loss(loss)
		test_acc(t, preds)

		return preds


	'''
	Build model
	'''

	network = Vgg16(10)
	input_layer = Input(shape=(28,28,1))
	output_layer = network(input_layer)
	model = Model(inputs=input_layer,outputs=output_layer)
	model.build(input_shape=(None,28,28,1))
	criterion = tf.losses.MeanSquaredError()
	optimizer = tf.keras.optimizers.Adam()


	'''
	Load data
	'''

	mnist = tf.keras.datasets.fashion_mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
	x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
	y_train = np.eye(10)[y_train].astype(np.float32)
	y_test = np.eye(10)[y_test].astype(np.float32)


	'''
	Train model
	'''

	epochs = 5
	batch_size = 2
	n_batches = x_train.shape[0] // batch_size

	train_loss = tf.keras.metrics.Mean()
	train_acc = tf.keras.metrics.CategoricalAccuracy()
	test_loss = tf.keras.metrics.Mean()
	test_acc = tf.keras.metrics.CategoricalAccuracy()


	for epoch in range(epochs):

		_x_train, _y_train = shuffle(x_train, y_train)

		for batch in range(n_batches):
			start = batch * batch_size
			end = start + batch_size
			train_step(_x_train[start:end], _y_train[start:end])
		preds = test_step(x_test, y_test)
		print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
			epoch+1,
			test_loss.result(),
			test_acc.result()
		))






