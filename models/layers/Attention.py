import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class Attention(Layer):
    '''
    Reference:
        Effective Approaches to Attention-based Neural Machine Translation
        https://arxiv.org/pdf/1508.04025.pdf
    '''
    def __init__(self,
                 output_dim,
                 hidden_dim,  # suppose dim(hs) = dim(ht)
                 **kwargs):
        self.supports_masking = True
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W_a = self.add_weight(name='W_a',
                                   shape=(self.hidden_dim,
                                          self.hidden_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)

        self.W_c = self.add_weight(name='W_c',
                                   shape=(self.hidden_dim + self.hidden_dim,
                                          self.output_dim),
                                   initializer='glorot_uniform',
                                   trainable=True)

        self.b = self.add_weight(name='b',
                                 shape=(self.output_dim),
                                 initializer='zeros',
                                 trainable=True)

        super().build(input_shape)

    def call(self, ht, hs, source=None, pad_value=0, mask=None):
        score = tf.einsum('ijk,kl->ijl', hs, self.W_a)
        score = tf.einsum('ijk,ilk->ijl', ht, score)

        score = score - tf.reduce_max(score,
                                      axis=-1,
                                      keepdims=True)  # softmax max trick

        if source is not None:
            len_source_sequences = \
                tf.reduce_sum(tf.cast(tf.not_equal(source, pad_value),
                                      tf.int32), axis=1)
            mask_source = \
                tf.cast(tf.sequence_mask(len_source_sequences,
                                         tf.shape(score)[-1]),
                        tf.float32)

            score = tf.exp(score) * mask_source[:, tf.newaxis, :]
        else:
            score = tf.exp(score)

        a = score / tf.reduce_sum(score, axis=-1, keepdims=True)
        c = tf.einsum('ijk,ikl->ijl', a, hs)

        h = tf.concat([c, ht], -1)
        return tf.nn.tanh(tf.einsum('ijk,kl->ijl', h, self.W_c) + self.b)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)

    def compute_mask(self, inputs, mask):
        return mask
