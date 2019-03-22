import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers \
    import Activation, Dense, Dropout, Embedding, Conv1D
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils.datasets.small_parallel_enja import load_small_parallel_enja
from utils.preprocessing.sequence import sort
from sklearn.utils import shuffle
from layers import PositionalEncoding
from layers import MultiHeadAttention
from layers import LayerNormalization


class Transformer(Model):
    def __init__(self,
                 depth_source,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128,
                 bos_value=1):
        super().__init__()
        self.encoder = Encoder(depth_source,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               max_len=max_len)
        self.decoder = Decoder(depth_target,
                               N=N,
                               h=h,
                               d_model=d_model,
                               d_ff=d_ff,
                               p_dropout=p_dropout,
                               max_len=max_len)
        self.out = Dense(depth_target, activation='softmax')
        self._max_len = max_len
        self._BOS = bos_value

    def call(self, source, target=None):
        source_mask = self.sequence_mask(source)

        hs = self.encoder(source, mask=source_mask)

        if target is not None:
            target = target[:, :-1]
            len_target_sequences = target.shape[1]
            target_mask = self.sequence_mask(target)[:, tf.newaxis, :]
            subsequent_mask = self.subsequence_mask(target)
            target_mask = tf.greater(target_mask + subsequent_mask, 1)

            y = self.decoder(target, hs,
                             mask=target_mask,
                             source_mask=source_mask)
            output = self.out(y)
        else:
            batch_size = source.shape[0]
            len_target_sequences = self._max_len

            output = tf.ones((batch_size, 1), dtype=tf.int32) * self._BOS

            for t in range(len_target_sequences - 1):
                target_mask = self.subsequence_mask(output)
                out = self.decoder(output, hs,
                                   mask=target_mask,
                                   source_mask=source_mask)
                out = self.out(out)[:, -1:, :]
                out = tf.argmax(out, axis=-1, output_type=tf.int32)
                output = tf.concat([output, out], axis=-1)

        return output

    def sequence_mask(self, x):
        len_sequences = \
            tf.reduce_sum(tf.cast(tf.not_equal(x, 0),
                                  tf.int32), axis=1)
        mask = \
            tf.cast(tf.sequence_mask(len_sequences,
                                     tf.shape(x)[-1]), tf.float32)
        return mask

    def subsequence_mask(self, x):
        shape = (x.shape[1], x.shape[1])
        mask = np.tril(np.ones(shape, dtype=np.int32), k=0)
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        return tf.tile(mask[tf.newaxis, :, :], [x.shape[0], 1, 1])


class Encoder(Model):
    def __init__(self,
                 depth_source,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128):
        super().__init__()
        self.embedding = Embedding(depth_source,
                                   d_model, mask_zero=True)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.encs = [EncoderLayer(h=h,
                                  d_model=d_model,
                                  d_ff=d_ff,
                                  p_dropout=p_dropout,
                                  max_len=max_len) for _ in range(N)]

    def call(self, x, mask=None):
        y = self.embedding(x)
        y = self.pe(y)
        for enc in self.encs:
            y = enc(y, mask=mask)

        return y


class EncoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128):
        super().__init__()
        self.attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()

    def call(self, x, mask=None):
        h = self.attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        y = self.ff(h)
        y = self.dropout2(y)
        y = self.norm2(h + y)

        return y

    def compute_mask(self, inputs, mask):
        return mask


class Decoder(Model):
    def __init__(self,
                 depth_target,
                 N=6,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128):
        super().__init__()
        self.embedding = Embedding(depth_target,
                                   d_model, mask_zero=True)
        self.pe = PositionalEncoding(d_model, max_len=max_len)
        self.decs = [DecoderLayer(h=h,
                                  d_model=d_model,
                                  d_ff=d_ff,
                                  p_dropout=p_dropout,
                                  max_len=max_len) for _ in range(N)]

    def call(self, x, hs,
             mask=None,
             source_mask=None):
        y = self.embedding(x)
        y = self.pe(y)

        for dec in self.decs:
            y = dec(y, hs,
                    mask=mask,
                    source_mask=source_mask)

        return y


class DecoderLayer(Model):
    def __init__(self,
                 h=8,
                 d_model=512,
                 d_ff=2048,
                 p_dropout=0.1,
                 max_len=128):
        super().__init__()
        self.self_attn = MultiHeadAttention(h, d_model)
        self.dropout1 = Dropout(p_dropout)
        self.norm1 = LayerNormalization()
        self.src_tgt_attn = MultiHeadAttention(h, d_model)
        self.dropout2 = Dropout(p_dropout)
        self.norm2 = LayerNormalization()
        self.ff = FFN(d_model, d_ff)
        self.dropout3 = Dropout(p_dropout)
        self.norm3 = LayerNormalization()

    def call(self, x, hs,
             mask=None,
             source_mask=None):
        h = self.self_attn(x, x, x, mask=mask)
        h = self.dropout1(h)
        h = self.norm1(x + h)

        z = self.src_tgt_attn(h, hs, hs,
                              mask=source_mask)
        z = self.dropout2(z)
        z = self.norm2(h + z)

        y = self.ff(z)
        y = self.dropout3(y)
        y = self.norm3(z + y)

        return y

    def compute_mask(self, inputs, mask):
        return mask


class FFN(Model):
    '''
    Position-wise Feed-Forward Networks
    '''
    def __init__(self, d_model, d_ff):
        super().__init__()
        # self.l1 = Dense(d_ff, activation='linear')
        # self.l2 = Dense(d_model, activation='linear')
        self.l1 = Conv1D(d_ff, 1, activation='linear')
        self.l2 = Conv1D(d_model, 1, activation='linear')

    def call(self, x):
        x = self.l1(x)
        x = tf.nn.relu(x)
        y = self.l2(x)
        return y


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    @tf.function
    def compute_loss(label, pred):
        return criterion(label, pred)

    # @tf.function
    def train_step(x, t, depth_t, pad_value=0):
        with tf.GradientTape() as tape:
            preds = model(x, t)
            t = t[:, 1:]
            label = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
            mask_t = tf.cast(tf.not_equal(t, pad_value), tf.float32)
            label = label * mask_t[:, :, tf.newaxis]
            loss = compute_loss(label, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return preds

    # @tf.function
    def valid_step(x, t, depth_t, pad_value=0):
        preds = model(x, t)
        t = t[:, 1:]
        label = tf.one_hot(t, depth=depth_t, dtype=tf.float32)
        mask_t = tf.cast(tf.not_equal(t, pad_value), tf.float32)
        label = label * mask_t[:, :, tf.newaxis]
        loss = compute_loss(label, preds)
        valid_loss(loss)

        return preds

    # @tf.function
    def test_step(x):
        preds = model(x)
        return preds

    def ids_to_sentence(ids, i2w):
        return [i2w[id] for id in ids]

    '''
    Load data
    '''
    class ParallelDataLoader(object):
        def __init__(self, dataset,
                     batch_size=128,
                     shuffle=False,
                     random_state=None):
            if type(dataset) is not tuple:
                raise ValueError('argument `dataset` must be tuple,'
                                 ' not {}.'.format(type(dataset)))
            self.dataset = list(zip(dataset[0], dataset[1]))
            self.batch_size = batch_size
            self.shuffle = shuffle
            if random_state is None:
                random_state = np.random.RandomState(1234)
            self.random_state = random_state
            self._idx = 0

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            return self

        def __next__(self):
            if self._idx >= len(self.dataset):
                self._reorder()
                raise StopIteration()

            x, y = zip(*self.dataset[self._idx:(self._idx + self.batch_size)])
            x, y = sort(x, y, order='descend')
            x = pad_sequences(x, padding='post')
            y = pad_sequences(y, padding='post')

            x = tf.convert_to_tensor(x, dtype=tf.int32)
            y = tf.convert_to_tensor(y, dtype=tf.int32)

            self._idx += self.batch_size

            return x, y

        def _reorder(self):
            if self.shuffle:
                self.data = shuffle(self.dataset,
                                    random_state=self.random_state)
            self._idx = 0

    (x_train, y_train), \
        (x_test, y_test), \
        (num_x, num_y), \
        (w2i_x, w2i_y), (i2w_x, i2w_y) = \
        load_small_parallel_enja(to_ja=True)

    train_dataloader = ParallelDataLoader((x_train, y_train),
                                          shuffle=True)
    valid_dataloader = ParallelDataLoader((x_test, y_test))
    test_dataloader = ParallelDataLoader((x_test, y_test),
                                         batch_size=1,
                                         shuffle=True)

    '''
    Build model
    '''
    model = Transformer(num_x,
                        num_y,
                        N=3,
                        h=4,
                        d_model=128,
                        d_ff=256,
                        max_len=20)
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()

    '''
    Train model
    '''
    epochs = 20
    train_loss = tf.keras.metrics.Mean()
    valid_loss = tf.keras.metrics.Mean()

    for epoch in range(epochs):
        print('-' * 20)
        print('Epoch: {}'.format(epoch+1))

        for idx, (source, target) in enumerate(train_dataloader):
            train_step(source, target, num_y)

        for (source, target) in valid_dataloader:
            valid_step(source, target, num_y)

        print('Valid loss: {:.3}'.format(valid_loss.result()))

        for idx, (source, target) in enumerate(test_dataloader):
            out = test_step(source)
            out = tf.reshape(out, shape=[-1]).numpy().tolist()
            out = ' '.join(ids_to_sentence(out, i2w_y))
            source = tf.reshape(source, shape=[-1]).numpy().tolist()
            target = tf.reshape(target, shape=[-1]).numpy().tolist()
            source = ' '.join(ids_to_sentence(source, i2w_x))
            target = ' '.join(ids_to_sentence(target, i2w_y))
            print('>', source)
            print('=', target)
            print('<', out)
            print()

            if idx >= 10:
                break
