import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Input, GlobalAveragePooling2D, Add, Dense, Activation, \
    BatchNormalization, Conv2D, MaxPooling2D


class ResNet34(object):
    '''
    Reference:
        "Deep Residual Learning for Image Recognition"
        https://arxiv.org/abs/1512.03385
    '''
    def __init__(self, input_shape, output_dim):
        x = Input(shape=input_shape)
        h = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(h)
        h = self._building_block(h, channel_out=64)
        h = self._building_block(h, channel_out=64)
        h = self._building_block(h, channel_out=64)
        h = Conv2D(128, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=128)
        h = self._building_block(h, channel_out=128)
        h = self._building_block(h, channel_out=128)
        h = self._building_block(h, channel_out=128)
        h = Conv2D(256, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = self._building_block(h, channel_out=256)
        h = Conv2D(512, kernel_size=(1, 1), strides=(2, 2))(h)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = self._building_block(h, channel_out=512)
        h = GlobalAveragePooling2D()(h)
        h = Dense(1000, activation='relu')(h)
        y = Dense(output_dim, activation='softmax')(h)
        self.model = Model(x, y)

    def __call__(self):
        return self.model

    def _building_block(self, x, channel_out=64):
        h = Conv2D(channel_out, kernel_size=(3, 3), padding='same')(x)
        h = BatchNormalization()(h)
        h = Activation('relu')(h)
        h = Conv2D(channel_out, kernel_size=(3, 3), padding='same')(h)
        h = BatchNormalization()(h)
        h = Add()([h, x])
        return Activation('relu')(h)


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

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
    resnet = ResNet34((28, 28, 1), 10)
    model = resnet()
    # model.summary()
    model.compile('adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    '''
    Train model
    '''
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
