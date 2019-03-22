import os
import copy
from collections import deque
import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense


class DQN(Model):
    '''
    Simple Deep Q-Network for CartPole
    '''
    def __init__(self):
        super().__init__()
        self.original = Network()
        self.target = Network()

    def call(self, x):
        return self.original(x)

    def q_original(self, x):
        return self.call(x)

    def q_target(self, x):
        return self.target(x)

    def copy_original(self):
        self.target = copy.deepcopy(self.original)


class Network(Model):
    def __init__(self):
        super().__init__()

        self.l1 = Dense(16, activation='relu')
        self.l2 = Dense(32, activation='relu')
        self.l3 = Dense(16, activation='relu')
        self.l4 = Dense(2, activation='linear')

    def call(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        y = self.l4(x)

        return y


class ReplayMemory(object):
    def __init__(self,
                 memory_size=50000):
        self.memory_size = memory_size
        self.memories = deque([], maxlen=memory_size)

    def append(self, memory):
        self.memories.append(memory)

    def sample(self, batch_size=128):
        indices = \
            np.random.permutation(range(len(self.memories)))[:batch_size]\
            .tolist()

        state = np.array([self.memories[i].state for i in indices])
        action = np.array([self.memories[i].action for i in indices])
        next_state = \
            np.array([self.memories[i].next_state for i in indices])
        reward = np.array([self.memories[i].reward for i in indices])
        terminal = np.array([self.memories[i].terminal for i in indices])

        return Memory(
            tf.convert_to_tensor(state, dtype=tf.float32),
            tf.convert_to_tensor(action, dtype=tf.float32),
            tf.convert_to_tensor(next_state, dtype=tf.float32),
            tf.convert_to_tensor(reward, dtype=tf.float32),
            tf.convert_to_tensor(terminal, dtype=tf.float32),
        )


class Memory(object):
    def __init__(self,
                 state,
                 action,
                 next_state,
                 reward,
                 terminal):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.terminal = terminal


class Epsilon(object):
    def __init__(self,
                 init=1.0,
                 end=0.1,
                 steps=10000):
        self.init = init
        self.end = end
        self.steps = steps

    def __call__(self, step):
        return max(0.1,
                   self.init + (self.end - self.init) / self.steps * step)


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)
    logger = tf.get_logger()
    logger.level = 40

    @tf.function
    def compute_loss(label, pred):
        return criterion(label, pred)

    # @tf.function
    def train_step(state, action, t):
        with tf.GradientTape() as tape:
            q_original = model(state)
            action = tf.one_hot(tf.cast(action, tf.int32), 2)
            q = tf.reduce_max(q_original * action, axis=1)
            loss = compute_loss(t, q)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        train_loss(loss)

        return loss

    '''
    Load env
    '''
    env = gym.make('CartPole-v0')

    '''
    Build model
    '''
    model = DQN()
    criterion = tf.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

    '''
    Build ReplayMemory
    '''
    initial_memory_size = 500
    replay_memory = ReplayMemory()

    step = 0
    while True:
        state = env.reset()
        terminal = False

        while not terminal:
            action = env.action_space.sample()
            next_state, reward, terminal, _ = env.step(action)
            memory = Memory(state, action, next_state, reward, int(terminal))
            replay_memory.append(memory)
            state = next_state
            step += 1

        if step >= initial_memory_size:
            break

    '''
    Train model
    '''
    n_episodes = 300
    gamma = 0.99
    step = 0
    copy_original_every = 1000
    eps = Epsilon()
    train_loss = tf.keras.metrics.Mean()

    model.copy_original()
    for episode in range(n_episodes):
        state = env.reset()
        terminal = False

        rewards = 0.
        q_max = []
        while not terminal:
            s = tf.constant(state[None], dtype=tf.float32)
            q = model.q_original(s)
            q_max.append(tf.reduce_max(q).numpy())

            # epsilon-greedy
            if np.random.random() < eps(step):
                action = env.action_space.sample()
            else:
                action = tf.argmax(q, axis=-1).numpy()[0]

            next_state, reward, terminal, _ = env.step(action)
            rewards += reward

            memory = Memory(state, action, next_state, reward, int(terminal))
            replay_memory.append(memory)

            sample = replay_memory.sample()
            q_target = model.q_target(sample.next_state)

            t = sample.reward \
                + (1 - sample.terminal) * gamma \
                * tf.reduce_max(q_target, axis=-1)

            train_step(sample.state, sample.action, t)

            state = next_state
            env.render()

            if (step + 1) % copy_original_every == 0:
                model.copy_original()

            step += 1

        template = 'Episode: {}, Reward: {}, Qmax: {:.3f}'
        print(template.format(
            episode+1,
            rewards,
            np.mean(q_max)
        ))

    env.close()
