import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets
from tensorflow.keras.models import Model
from tensorflow.keras.layers import \
    Dense, Activation, Reshape, BatchNormalization, \
    Conv2D, MaxPooling2D,  UpSampling2D, LeakyReLU
from tensorflow.keras import optimizers
from tensorflow.keras import metrics
from utils.datasets import DataLoader
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt


class GAN(Model):
    '''
    Simple Generative Adversarial Network
    '''
    def __init__(self):
        super().__init__()
        self.G = Generator()
        self.D = Discriminator()

    def call(self, x):
        x = self.G(x)
        y = self.D(x)

        return y


class Discriminator(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2D(128,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same')
        self.relu1 = LeakyReLU(0.2)
        self.conv2 = Conv2D(256,
                            kernel_size=(3, 3),
                            strides=(2, 2),
                            padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = LeakyReLU(0.2)
        self.reshape = Reshape([256*7*7])
        self.fc = Dense(1024)
        self.bn3 = BatchNormalization()
        self.relu3 = LeakyReLU(0.2)
        self.out = Dense(1, activation='sigmoid')

    def call(self, x):
        h = self.conv1(x)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.reshape(h)
        h = self.fc(h)
        h = self.bn3(h)
        h = self.relu3(h)
        y = self.out(h)

        return y


class Generator(Model):
    def __init__(self, input_dim=100):
        super().__init__()
        self.linear = Dense(256*14*14)
        self.bn1 = BatchNormalization()
        self.relu1 = Activation('relu')
        self.reshape = Reshape([14, 14, 256])
        self.upsample = UpSampling2D(size=(2, 2))
        self.conv1 = Conv2D(128,
                            kernel_size=(3, 3),
                            padding='same')
        self.bn2 = BatchNormalization()
        self.relu2 = Activation('relu')
        self.conv2 = Conv2D(64,
                            kernel_size=(3, 3),
                            padding='same')
        self.bn3 = BatchNormalization()
        self.relu3 = Activation('relu')
        self.conv3 = Conv2D(1,
                            kernel_size=(1, 1))
        self.out = Activation('sigmoid')

    def call(self, x):
        h = self.linear(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.reshape(h)
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = self.relu2(h)
        h = self.conv2(h)
        h = self.bn3(h)
        h = self.relu3(h)
        h = self.conv3(h)
        y = self.out(h)

        return y


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    @tf.function
    def compute_loss(label, pred):
        return criterion(label, pred)

    @tf.function
    def train_step(x):
        train_step_D(x)
        train_step_G(x)

    @tf.function
    def train_step_D(x):
        batch_size = x.shape[0]

        t_r = tf.ones(batch_size, dtype=tf.float32)  # real
        t_f = tf.zeros(batch_size, dtype=tf.float32)  # fake

        with tf.GradientTape() as tape:
            # real images
            p_r = model.D(x)
            loss_D_real = compute_loss(t_r, p_r)

            # fake images
            noise = gen_noise(batch_size)
            z = model.G(noise)
            p_f = model.D(z)
            loss_D_fake = compute_loss(t_f, p_f)

            loss_D = loss_D_real + loss_D_fake

        grads = tape.gradient(loss_D, model.D.trainable_variables)
        optimizer_D.apply_gradients(zip(grads, model.D.trainable_variables))
        train_loss_D(loss_D)

    @tf.function
    def train_step_G(x):
        batch_size = x.shape[0]

        t = tf.ones(batch_size, dtype=tf.float32)

        with tf.GradientTape() as tape:
            noise = gen_noise(batch_size)
            z = model.G(noise)
            preds = model.D(z)
            loss_G = compute_loss(t, preds)
        grads = tape.gradient(loss_G, model.G.trainable_variables)
        optimizer_G.apply_gradients(zip(grads, model.G.trainable_variables))
        train_loss_G(loss_G)

    def generate(batch_size=10):
        noise = gen_noise(batch_size)
        gen = model.G(noise)

        return gen

    def gen_noise(batch_size):
        return tf.random.uniform([batch_size, 100], dtype=tf.float32)

    '''
    Load data
    '''
    mnist = datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)

    train_dataloader = DataLoader((x_train, y_train),
                                  batch_size=100,
                                  shuffle=True)

    '''
    Build model
    '''
    model = GAN()
    criterion = tf.losses.BinaryCrossentropy()
    optimizer_D = optimizers.Adam(lr=0.0002)
    optimizer_G = optimizers.Adam(lr=0.0002)

    '''
    Train model
    '''
    epochs = 100
    out_path = os.path.join(os.path.dirname(__file__),
                            '..', 'output')
    train_loss_D = metrics.Mean()
    train_loss_G = metrics.Mean()

    for epoch in range(epochs):

        for (x, _) in train_dataloader:
            train_step(x)

        print('Epoch: {}, D Cost: {:.3f}, G Cost {:.3f}'.format(
            epoch+1,
            train_loss_D.result(),
            train_loss_G.result()
        ))

        if epoch % 5 == 4 or epoch == epochs - 1:
            images = generate(batch_size=16)
            images = images.numpy().reshape(-1, 28, 28)
            plt.figure(figsize=(6, 6))
            for i, image in enumerate(images):
                plt.subplot(4, 4, i+1)
                plt.imshow(image, cmap='binary')
                plt.axis('off')
            plt.tight_layout()
            # plt.show()
            template = '{}/gan_fashion_mnist_epoch_{:0>4}.png'
            plt.savefig(template.format(out_path, epoch+1), dpi=300)
