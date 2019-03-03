import os
import subprocess
import numpy as np


'''
Download 50k En/Ja Parallel Corpus
from https://github.com/odashi/small_parallel_enja
and transform words to IDs.


Original Source from:

https://github.com/yusugomori/tftf/blob/master/tftf/datasets/small_parallel_enja.py
'''


def load_small_parallel_enja(path=None,
                             to_ja=True,
                             pad_value=0,
                             start_char=1,
                             end_char=2,
                             oov_char=3,
                             index_from=4,
                             pad='<PAD>',
                             bos='<BOS>',
                             eos='<EOS>',
                             oov='<UNK>',
                             add_bos=True,
                             add_eos=True):
    url_base = 'https://raw.githubusercontent.com/' \
               'odashi/small_parallel_enja/master/'

    path = path or 'small_parallel_enja'
    dir_path = os.path.join(os.path.expanduser('~'),
                            '.tftf', 'datasets', path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    f_ja = ['train.ja', 'test.ja']
    f_en = ['train.en', 'test.en']

    for f in (f_ja + f_en):
        f_path = os.path.join(dir_path, f)
        if not os.path.exists(f_path):
            url = url_base + f
            print('Downloading {}'.format(f))
            cmd = ['curl', '-o', f_path, url]
            subprocess.call(cmd)

    f_ja_train = os.path.join(dir_path, f_ja[0])
    f_test_ja = os.path.join(dir_path, f_ja[1])
    f_en_train = os.path.join(dir_path, f_en[0])
    f_test_en = os.path.join(dir_path, f_en[1])

    (ja_train, test_ja), num_words_ja, (w2i_ja, i2w_ja) = \
        _build(f_ja_train, f_test_ja,
               pad_value, start_char, end_char, oov_char, index_from,
               pad, bos, eos, oov, add_bos, add_eos)
    (en_train, test_en), num_words_en, (w2i_en, i2w_en) = \
        _build(f_en_train, f_test_en,
               pad_value, start_char, end_char, oov_char, index_from,
               pad, bos, eos, oov, add_bos, add_eos)

    if to_ja:
        x_train, x_test, num_X, w2i_X, i2w_X = \
            en_train, test_en, num_words_en, w2i_en, i2w_en
        y_train, y_test, num_y, w2i_y, i2w_y = \
            ja_train, test_ja, num_words_ja, w2i_ja, i2w_ja
    else:
        x_train, x_test, num_X, w2i_X, i2w_X = \
            ja_train, test_ja, num_words_ja, w2i_ja, i2w_ja
        y_train, y_test, num_y, w2i_y, i2w_y = \
            en_train, test_en, num_words_en, w2i_en, i2w_en

    x_train, x_test = np.array(x_train), np.array(x_test)
    y_train, y_test = np.array(y_train), np.array(y_test)

    return (x_train, y_train), (x_test, y_test), \
        (num_X, num_y), (w2i_X, w2i_y), (i2w_X, i2w_y)


def _build(f_train, f_test,
           pad_value=0,
           start_char=1,
           end_char=2,
           oov_char=3,
           index_from=4,
           pad='<PAD>',
           bos='<BOS>',
           eos='<EOS>',
           oov='<UNK>',
           add_bos=True,
           add_eos=True):

    builder = _Builder(pad_value=pad_value,
                       start_char=start_char,
                       end_char=end_char,
                       oov_char=oov_char,
                       index_from=index_from,
                       pad=pad,
                       bos=bos,
                       eos=eos,
                       oov=oov,
                       add_bos=add_bos,
                       add_eos=add_eos)
    builder.fit(f_train)
    train = builder.transform(f_train)
    test = builder.transform(f_test)

    return (train, test), builder.num_words, (builder.w2i, builder.i2w)


class _Builder(object):
    def __init__(self,
                 pad_value=0,
                 start_char=1,
                 end_char=2,
                 oov_char=3,
                 index_from=4,
                 pad='<PAD>',
                 bos='<BOS>',
                 eos='<EOS>',
                 oov='<UNK>',
                 add_bos=True,
                 add_eos=True):
        self._vocab = None
        self._w2i = None
        self._i2w = None

        self.pad_value = pad_value
        self.start_char = start_char
        self.end_char = end_char
        self.oov_char = oov_char
        self.index_from = index_from
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self.oov = oov

        self.add_bos = add_bos
        self.add_eos = add_eos

    @property
    def num_words(self):
        return max(self._w2i.values()) + 1

    @property
    def w2i(self):
        '''
        Dict of word to index
        '''
        return self._w2i

    @property
    def i2w(self):
        '''
        Dict of index to word
        '''
        return self._i2w

    def fit(self, f_path):
        self._vocab = set()
        self._w2i = {}
        for line in open(f_path, encoding='utf-8'):
            _sentence = line.strip().split()
            self._vocab.update(_sentence)

        self._w2i = {w: (i + self.index_from)
                     for i, w in enumerate(self._vocab)}
        if self.pad_value >= 0:
            self._w2i[self.pad] = self.pad_value
        self._w2i[self.bos] = self.start_char
        self._w2i[self.eos] = self.end_char
        self._w2i[self.oov] = self.oov_char
        self._i2w = {i: w for w, i in self._w2i.items()}

    def transform(self, f_path):
        if self._vocab is None or self._w2i is None:
            raise AttributeError('`{}.fit` must be called before `transform`.'
                                 ''.format(self.__class__.__name__))
        sentences = []
        for line in open(f_path, encoding='utf-8'):
            _sentence = line.strip().split()
            # _sentence = [self.bos] + _sentence + [self.eos]
            if self.add_bos:
                _sentence = [self.bos] + _sentence
            if self.add_eos:
                _sentence = _sentence + [self.eos]
            sentences.append(self._encode(_sentence))
        return sentences

    def _encode(self, sentence):
        encoded = []
        for w in sentence:
            if w not in self._w2i:
                id = self.oov_char
            else:
                id = self._w2i[w]
            encoded.append(id)

        return encoded
