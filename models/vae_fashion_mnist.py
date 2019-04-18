import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from utils.datasets import DataLoader
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class VAE(Model):
    '''
    Simple Variational Autoencoder
    '''
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, x):
        mean, var = self.encoder(x)
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        return y

    def reparameterize(self, mean, var):
        eps = tf.random.normal(mean.shape)
        z = mean + tf.math.sqrt(var) * eps
        return z

    def lower_bound(self, x):
        mean, var = self.encoder(x)
        kl = - 1/2 * tf.reduce_mean(tf.reduce_sum(1
                                                  + self._log(var, max=var)
                                                  - mean**2
                                                  - var,
                                                  axis=1))
        z = self.reparameterize(mean, var)
        y = self.decoder(z)

        reconst = tf.reduce_mean(tf.reduce_sum(x * self._log(y)
                                               + (1 - x) * self._log(1 - y),
                                               axis=1))

        L = reconst - kl

        return L

    def _log(self, value, min=1.e-10, max=1.0):
        return tf.math.log(tf.clip_by_value(value, min, max))


class Encoder(Model):
    def __init__(self):
        super().__init__()
        self.l1 = Dense(200, activation='relu')
        self.l2 = Dense(200, activation='relu')
        self.l_mean = Dense(10, activation='linear')
        self.l_var = Dense(10, activation=tf.nn.softplus)

    def call(self, x):
        h = self.l1(x)
        h = self.l2(h)

        mean = self.l_mean(h)
        var = self.l_var(h)

        return mean, var


class Decoder(Model):
    def __init__(self):
        super().__init__()
        self.l1 = Dense(200, activation='relu')
        self.l2 = Dense(200, activation='relu')
        self.out = Dense(784, activation='sigmoid')

    def call(self, x):
        h = self.l1(x)
        h = self.l2(h)
        y = self.out(h)

        return y


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    @tf.function
    def compute_loss(x):
        return -1 * criterion(x)

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            loss = compute_loss(x)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

    def generate(batch_size=16):
        z = gen_noise(batch_size)
        gen = model.decoder(z)
        gen = tf.reshape(gen, [-1, 28, 28])

        return gen

    def gen_noise(batch_size):
        return tf.random.normal([batch_size, 10])

    '''
    Load data
    '''
    mnist = datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)

    train_dataloader = DataLoader((x_train, y_train),
                                  batch_size=100,
                                  shuffle=True)

    '''
    Build model
    '''
    model = VAE()
    criterion = model.lower_bound
    optimizer = optimizers.Adam()

    '''
    Train model
    '''
    epochs = 10
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'output')
    train_loss = metrics.Mean()

    for epoch in range(epochs):

        for (x, _) in train_dataloader:
            train_step(x)

        print('Epoch: {}, Cost: {:.3f}'.format(
            epoch+1,
            train_loss.result()
        ))

        if epoch % 5 == 4 or epoch == epochs - 1:
            images = generate(batch_size=16)
            images = images.numpy()
            plt.figure(figsize=(6, 6))
            for i, image in enumerate(images):
                plt.subplot(4, 4, i+1)
                plt.imshow(image, cmap='binary')
                plt.axis('off')
            plt.tight_layout()
            # plt.show()
            template = '{}/vae_fashion_mnist_epoch_{:0>4}.png'
            plt.savefig(template.format(out_path, epoch+1), dpi=300)
