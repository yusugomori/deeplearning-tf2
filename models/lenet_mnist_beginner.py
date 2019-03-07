import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    '''
    Load data
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = Sequential([
        Conv2D(6, kernel_size=(5, 5),
               padding='valid', activation='relu'),
        MaxPooling2D(padding='same'),
        Conv2D(16, kernel_size=(5, 5),
               padding='valid', activation='relu'),
        MaxPooling2D(padding='same'),
        Flatten(),
        Dense(120, activation='relu'),
        Dense(84, activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile('adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])

    '''
    Train model
    '''
    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)
