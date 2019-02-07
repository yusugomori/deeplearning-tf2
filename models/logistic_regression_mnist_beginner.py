import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


if __name__ == '__main__':
    np.random.seed(1234)

    '''
    Load data
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = Sequential([
        Dense(200, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile('adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    '''
    Train model
    '''
    print(model.trainable_weights)
    # model.fit(x_train, y_train, epochs=5)
    # model.evaluate(x_test, y_test)
