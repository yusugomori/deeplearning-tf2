import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer


class DotProductAttention(Layer):
    def __init__(self,
                 d_model,
                 **kwargs):
        self.d_model = d_model
        super().__init__(**kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, q, k, v, mask=None):
        score = tf.einsum('ijk,ilk->ijl', q, k)
        score = score - tf.reduce_max(score,
                                      axis=-1,
                                      keepdims=True)  # softmax max trick
        score = tf.exp(score)
        if mask is not None:
            # suppose `mask` is a mask of source
            # in source-target-attention, source is `k` and `v`
            if len(mask.shape) == 2:
                mask = mask[:, tf.newaxis, :]
            mask = tf.cast(mask, tf.float32)
            score = score * mask

        a = score / tf.reduce_sum(score, axis=-1, keepdims=True)
        c = tf.einsum('ijk,ikl->ijl', a, v)

        return c

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.d_model)

    def compute_mask(self, inputs, mask):
        return mask
