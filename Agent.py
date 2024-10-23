# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 19:43:27 2024

@author: Archanaa N
"""

import numpy as np
from tensorflow import keras
from collections import deque
import random
 
class DQNAgent:
    def __init__(self, state_size, action_size, num_products):
        self.state_size = state_size
        self.action_size = action_size
        self.num_products = num_products
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99  # Speed up decay
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
 
    def _build_model(self):
        model = keras.Sequential([
            keras.layers.Dense(64, input_dim=self.state_size, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model
 
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
 
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
 
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.uniform(10, 100, self.num_products)
        act_values = self.model.predict(state.reshape(1, -1))
        return self._decode_action(np.argmax(act_values[0]))
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([sample[0] for sample in minibatch])
        actions = [sample[1] for sample in minibatch]
        rewards = [sample[2] for sample in minibatch]
        next_states = np.array([sample[3] for sample in minibatch])
        dones = [sample[4] for sample in minibatch]
 
    # Predict Q-values for next states
        target_q_values_next = self.target_model.predict(next_states)
 
    # Predict Q-values for current states
        target_q_values = self.model.predict(states)
 
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(target_q_values_next[i])
            action_index = self._encode_action(actions[i])
            target_q_values[i][action_index] = target
 
    # Train the model on the entire batch
        self.model.fit(states, target_q_values, epochs=3, verbose=0)
 
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
 
    def _encode_action(self, action):
        encoded = 0
        for i, price in enumerate(action):
            encoded += int((price - 10) / 10) * (9 ** i)
        return encoded
 
    def _decode_action(self, encoded_action):
        action = np.zeros(self.num_products)
        for i in range(self.num_products):
            action[i] = (encoded_action % 9) * 10 + 10
            encoded_action //= 9
        return action
 
    def save(self, filename):
        self.model.save_weights(filename + ".weights.h5")
 
    def load(self, filename):
        self.model.load_weights(filename + ".weights.h5")
        
        