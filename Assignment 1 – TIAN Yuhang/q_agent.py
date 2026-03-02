import numpy as np
import random
from config import ALPHA, GAMMA, EPSILON, EPSILON_DECAY, MIN_EPSILON

class QLearningAgent:
    def __init__(self, state_size=256, action_size=3, alpha=ALPHA, gamma=GAMMA,
                 epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, min_epsilon=MIN_EPSILON):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.Q = np.zeros((state_size, action_size))

    def choose_action(self, state, training=True):
        # Select actions based on the epsilon-greedy strategy
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.Q[state])

    def update(self, state, action, reward, next_state, done):
        # Update Q-value
        current_q = self.Q[state][action]
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(self.Q[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.Q[state][action] = new_q

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)