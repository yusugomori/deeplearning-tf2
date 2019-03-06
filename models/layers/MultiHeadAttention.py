import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Layer
from .ScaledDotProductAttention import ScaledDotProductAttention


class MultiHeadAttention(Layer):
    def __init__(self,
                 h,
                 d_model,
                 **kwargs):
        self.h = h
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.attn = ScaledDotProductAttention(d_model)

        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W_q = self.add_weight(name='W_q',
                                   shape=(self.h,
                                          self.d_model,
                                          self.d_k),
                                   initializer='glorot_normal',
                                   trainable=True)

        self.W_k = self.add_weight(name='W_k',
                                   shape=(self.h,
                                          self.d_model,
                                          self.d_k),
                                   initializer='glorot_normal',
                                   trainable=True)

        self.W_v = self.add_weight(name='W_v',
                                   shape=(self.h,
                                          self.d_model,
                                          self.d_v),
                                   initializer='glorot_normal',
                                   trainable=True)

        self.W_o = self.add_weight(name='W_o',
                                   shape=(self.h * self.d_v,
                                          self.d_model),
                                   initializer='glorot_normal',
                                   trainable=True)

        self.b_o = self.add_weight(name='b_o',
                                   shape=(self.d_model),
                                   initializer='zeros',
                                   trainable=True)

        super().build(input_shape)

    def call(self, q, k, v, mask=None):
        q = tf.einsum('hijk,hkl->hijl',
                      tf.tile(q[tf.newaxis, :, :, :],
                              [self.h, 1, 1, 1]),
                      self.W_q)
        k = tf.einsum('hijk,hkl->hijl',
                      tf.tile(k[tf.newaxis, :, :, :],
                              [self.h, 1, 1, 1]),
                      self.W_k)
        v = tf.einsum('hijk,hkl->hijl',
                      tf.tile(v[tf.newaxis, :, :, :],
                              [self.h, 1, 1, 1]),
                      self.W_v)

        q = tf.reshape(q, shape=(-1, q.shape[-2], q.shape[-1]))
        k = tf.reshape(k, shape=(-1, k.shape[-2], k.shape[-1]))
        v = tf.reshape(v, shape=(-1, v.shape[-2], v.shape[-1]))

        if mask is not None:
            multiples = [self.h] + [1] * (len(mask.shape) - 1)
            mask = tf.tile(mask, multiples=multiples)

        c = self.attn(q, k, v, mask=mask)
        c = tf.split(c, self.h, axis=0)
        c = tf.concat(c, axis=-1)

        out = tf.einsum('ijk,kl->ijl', c, self.W_o) + self.b_o

        return out

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.h * self.d_v)

    def compute_mask(self, input, mask):
        return mask
