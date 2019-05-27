import abc
from collections import deque
import numpy as np
import random


class QLearningAgent:
    def __init__(self, state_size, action_size, gamma=0.95, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1, batch_size=32):
        self.state_size = state_size
        self.action_size = action_size

        # hyperparameters
        self.gamma = gamma  # discount rate on future rewards
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay  # the decay of epsilon after each training batch
        self.epsilon_min = epsilon_min  # the minimum exploration rate permissible
        self.batch_size = batch_size  # maximum size of the batches sampled from memory

        # agent state
        self.model = self.build_model()
        self.memory = deque(maxlen=2000)

    @abc.abstractmethod
    def build_model(self):
        return None

    def select_action(self, state, do_train=True):
        if do_train and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def record(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return 0

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            #self.model.fit(state, target_f, epochs=1, verbose=0)
            # For changes in versions
            self.model.fit(state, target_f, nb_epoch=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay