import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class ScaledDotProductAttention(Layer):
    def __init__(self,
                 d_model,
                 **kwargs):
        self.supports_masking = True
        self.d_model = d_model
        self.scaler = np.sqrt(d_model)
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, q, k, v, mask=None):
        score = tf.einsum('ijk,ilk->ijl', q, k) / self.scaler
        score = score - tf.reduce_max(score,
                                      axis=-1,
                                      keepdims=True)  # softmax max trick
        score = tf.exp(score)
        if mask is not None:
            # suppose `mask` is a mask of source
            mask = tf.cast(mask, tf.float32)
            score *= mask[:, tf.newaxis, :]

        a = score / tf.reduce_sum(score, axis=-1, keepdims=True)
        c = tf.einsum('ijk,ikl->ijl', a, v)

        return c

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

    def compute_mask(self, inputs, mask):
        return mask
