import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    GlobalAveragePooling2D, Add, \
    Dense, Activation, Flatten, \
    BatchNormalization, Conv2D, MaxPooling2D
from sklearn.utils import shuffle


class ResNet50(Model):
    '''
    Reference:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    '''
    def __init__(self, input_shape, output_dim):
        super().__init__()
        self.conv1 = Conv2D(64, input_shape=input_shape,
                            kernel_size=(7, 7),
                            strides=(2, 2),
                            padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.pool1 = MaxPooling2D(pool_size=(3, 3),
                                  strides=(2, 2),
                                  padding='same')
        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = [
            self._building_block(256) for _ in range(2)
        ]
        self.conv2 = Conv2D(512,
                            kernel_size=(1, 1),
                            strides=(2, 2))
        self.block2 = [
            self._building_block(512) for _ in range(4)
        ]
        self.conv3 = Conv2D(1024,
                            kernel_size=(1, 1),
                            strides=(2, 2))
        self.block3 = [
            self._building_block(1024) for _ in range(6)
        ]
        self.conv4 = Conv2D(2048,
                            kernel_size=(1, 1),
                            strides=(2, 2))
        self.block4 = [
            self._building_block(2048) for _ in range(3)
        ]
        self.avg_pool = GlobalAveragePooling2D()
        self.fc = Dense(1000, activation='relu')
        self.out = Dense(output_dim, activation='softmax')

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.pool1(h)
        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc(h)
        y = self.out(h)
        return y

    def _building_block(self,
                        channel_out=64,
                        channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return Block(channel_in, channel_out)


class Block(Model):
    def __init__(self, channel_in=64, channel_out=256):
        super().__init__()
        channel = channel_out // 4
        self.conv1 = Conv2D(channel,
                            kernel_size=(1, 1),
                            padding='same')
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.conv2 = Conv2D(channel,
                            kernel_size=(3, 3),
                            padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = BatchNormalization()
        self.conv3 = Conv2D(channel_out,
                            kernel_size=(1, 1),
                            padding='same')
        self.bn3 = BatchNormalization()
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.add = Add()
        self.relu3 = Activation('relu')

    def call(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv3(h)
        h = self.bn3(h)
        shortcut = self.shortcut(x)
        h = self.add([h, shortcut])
        y = self.relu3(h)
        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_out):
        return Conv2D(channel_out,
                      kernel_size=(1, 1),
                      padding='same')


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
    model = ResNet50((28, 28, 1), 10)
    model.build(input_shape=(None, 28, 28, 1))
    # model.summary()
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
