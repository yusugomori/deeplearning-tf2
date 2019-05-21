import tensorflow as tf 

class Vgg16(tf.keras.Model):
	def __init__(self,output_nodes):
		super(Vgg16, self).__init__()
		# layers needed
		self.conv1_1 = tf.keras.layers.Conv2D(
					input_shape=[None,28,28,1],filters=64, kernel_size=3,
					padding="same", activation="relu")
		self.conv1_2 = tf.keras.layers.Conv2D(
					filters=64, kernel_size=3,
					padding="same", activation="relu")
		self.conv2_1 = tf.keras.layers.Conv2D(
					filters=128, kernel_size=3,
					padding="same", activation="relu")
		self.conv2_2 = tf.keras.layers.Conv2D(
					filters=128, kernel_size=3,
					padding="same", activation="relu")
		self.conv3_1 = tf.keras.layers.Conv2D(
					filters=256,kernel_size=3,
					padding="same", activation="relu")
		self.conv3_2 = tf.keras.layers.Conv2D(
					filters=256,kernel_size=3,
					padding="same", activation="relu")
		self.conv3_3 = tf.keras.layers.Conv2D(
					filters=256,kernel_size=3,
					padding="same", activation="relu")
		
		self.conv4_1 = tf.keras.layers.Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv4_2 = tf.keras.layers.Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv4_3 = tf.keras.layers.Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv5_1 = tf.keras.layers.Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv5_2 = tf.keras.layers.Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.conv5_3 = tf.keras.layers.Conv2D(
					filters=512, kernel_size=3,
					padding="same", activation="relu")
		self.dense1_1 = tf.keras.layers.Dense(
					units=4096, activation="relu")
		self.dense1_2 = tf.keras.layers.Dense(
					units=4096, activation="relu")
		self.dense2 = tf.keras.layers.Dense(
					units=output_nodes, activation="softmax")
		self.maxPool = tf.keras.layers.MaxPool2D(
					pool_size=2, strides=2, padding="same")
		self.flatten = tf.keras.layers.Flatten()

	def call(self,input):
		# ops 
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


model = Vgg16(10)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[..., tf.newaxis]
print(x_train.shape)

model.fit(x_train, y_train, epochs=5)
