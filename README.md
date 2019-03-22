# Deep Learning with TensorFlow 2.X

Implementations of neural network models with tf (>=2.0)

See also implementations with PyTorch 1.0 [here](https://github.com/yusugomori/deeplearning-pytorch).

## Requirements

* TensorFlow >= 2.0

```shell
$ pip install tensorflow==2.0.0-alpha0
```

or

```shell
$ pip install tf-nightly-2.0-preview
```

## Models

* Logistic Regression
* MLP
* LeNet
* ResNet (ResNet34, ResNet50)
* Encoder-Decoder (LSTM)
* EncoderDecoder (Attention)
* Transformer

```
models/
├── densenet121_cifar10_beginner.py
├── encoder_decoder_attention.py
├── encoder_decoder_lstm.py
├── lenet_mnist.py
├── lenet_mnist_beginner.py
├── logistic_regression_mnist.py
├── logistic_regression_mnist_beginner.py
├── mlp_mnist.py
├── mlp_mnist_beginner.py
├── resnet34_fashion_mnist.py
├── resnet34_fashion_mnist_beginner.py
├── resnet50_fashion_mnist.py
├── resnet50_fashion_mnist_beginner.py
└── transformer.py
|
└── layers/
    ├── Attention.py
    ├── DotProductAttention.py
    ├── LayerNormalization.py
    ├── MultiHeadAttention.py
    ├── PositionalEncoding.py
    └── ScaledDotProductAttention.py
```

*_beginner.py is the file using only Keras.
