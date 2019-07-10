import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Conv2D, Dense, MaxPool2D, Flatten, Input
from sklearn.utils import shuffle


class Vgg16(Model):
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.conv1_1 = Conv2D(64, input_shape=input_shape,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv1_2 = Conv2D(64,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same')
        self.conv2_1 = Conv2D(128,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv2_2 = Conv2D(128,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same')
        self.conv3_1 = Conv2D(256,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv3_2 = Conv2D(256,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv3_3 = Conv2D(256,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same')
        self.conv4_1 = Conv2D(512,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv4_2 = Conv2D(512,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv4_3 = Conv2D(512,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.pool4 = MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same')
        self.conv5_1 = Conv2D(512,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv5_2 = Conv2D(512,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.conv5_3 = Conv2D(512,
                              kernel_size=(3, 3),
                              padding='same',
                              activation='relu')
        self.pool5 = MaxPool2D(pool_size=(2, 2),
                               strides=(2, 2),
                               padding='same')
        self.flatten = Flatten()
        self.fc1 = Dense(4096, activation='relu')
        self.fc2 = Dense(4096, activation='relu')
        self.out = Dense(output_dim, activation='softmax')

    def call(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        h = self.pool3(h)
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        h = self.pool4(h)
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        h = self.pool5(h)
        h = self.flatten(h)
        h = self.fc1(h)
        h = self.fc2(h)
        y = self.out(h)
        return y


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

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
    Load data
    '''
    mnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = Vgg16((28, 28, 1), 10)
    model.build(input_shape=(None, 28, 28, 1))
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    '''
    Train model
    '''
    epochs = 5
    batch_size = 100
    n_batches = x_train.shape[0] // batch_size

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    test_acc = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(epochs):

        _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

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
