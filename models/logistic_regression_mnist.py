import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


class LogisticRegression(Model):
    def __init__(self):
        super().__init__()
        self.dense = Dense(200, activation='relu')
        self.out = Dense(10, activation='softmax')

    def call(self, x):
        x = self.dense(x)
        y = self.out(x)
        return y


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)

    @tf.function
    def compute_loss(label, pred, from_logits=False):
        return categorical_crossentropy(label, pred,
                                        from_logits=from_logits)

    @tf.function
    def train_step(x, t):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = compute_loss(t, preds)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)
        train_acc(t, preds)

        return preds

    @tf.function
    def test_step(x, t):
        preds = model(x)
        loss = compute_loss(t, preds)
        test_loss(loss)
        test_acc(t, preds)

        return preds

    '''
    Load data
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 784) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 784) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    '''
    Build model
    '''
    model = LogisticRegression()
    optimizer = tf.keras.optimizers.Adam()

    '''
    Train model
    '''
    epochs = 10
    batch_size = 100
    n_batches = x_train.shape[0] // batch_size

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    test_acc = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(epochs):

        _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            train_step(_x_train[start:end], _y_train[start:end])

        if epoch % 5 == 4 or epoch == epochs - 1:
            preds = test_step(x_test, y_test)
            print('Epoch: {}, Valid Cost: {:.3f}, Valid Acc: {:.3f}'.format(
                epoch+1,
                test_loss.result(),
                test_acc.result()
            ))
