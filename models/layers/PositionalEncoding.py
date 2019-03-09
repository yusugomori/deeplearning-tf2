import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    '''
    Positional encoding layer with sinusoid
    '''
    def __init__(self, output_dim,
                 max_len=6000,
                 **kwargs):
        self.output_dim = output_dim
        self.max_len = max_len
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.PE = self.add_weight(name='PE',
                                  shape=(self.max_len,
                                         self.output_dim),
                                  initializer=self.initializer,
                                  trainable=False,
                                  dtype=tf.float32)

        super().build(input_shape)

    def call(self, x, mask=None):
        pe = self.PE[tf.newaxis, :tf.shape(x)[1], :]
        if mask is not None:
            pe = tf.tile(pe, [mask.shape[0], 1, 1])
            pe *= tf.cast(mask, tf.float32)[:, :, tf.newaxis]
        return x + pe

    def initializer(self, input_shape, dtype=tf.float32):
        pe = \
            np.array([[pos / np.power(10000, 2 * (i // 2) / self.output_dim)
                       for i in range(self.output_dim)]
                      for pos in range(self.max_len)])

        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        return tf.convert_to_tensor(pe, dtype=tf.float32)

    def compute_mask(self, inputs, mask):
        return mask
