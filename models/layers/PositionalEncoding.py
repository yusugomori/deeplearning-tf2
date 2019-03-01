import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    '''
    Positional encoding layer with sinusoid
    '''
    def __init__(self, output_dim,
                 maxlen=6000,
                 **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.maxlen = maxlen
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.PE = self.add_weight(name='PE',
                                  shape=(self.maxlen,
                                         self.output_dim),
                                  initializer=self.initializer,
                                  trainable=False)

        super().build(input_shape)

    def call(self, x, mask=None):
        pe = self.PE[tf.newaxis, :tf.shape(x)[1], :]
        return x + pe

    def initializer(self, input_shape, dtype=tf.float32):
        pe = np.zeros(shape=input_shape, dtype=np.float32)
        pos = np.arange(0, self.maxlen)[:, np.newaxis]
        div = np.power(10000,
                       np.arange(0, self.output_dim, 2) / self.output_dim)
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)

        return pe

    def compute_mask(self, inputs, mask):
        return mask
