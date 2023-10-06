import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import random
import keras
from collections import deque
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import datetime
import os
from tensorflow.python.client import device_lib

# Deep Q Network off-policy

def named_logs(model, logs):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    return result

class DeepQNetwork:
    def __init__(self, state_size, action_size, checkpoint='None'):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99  
        self.epsilon = 1.0  
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        self.counter = 0
        if os.path.exists(checkpoint):
            self.model.load_weights(checkpoint)
            self.target_model.load_weights(checkpoint)


    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta
        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))

        model.add(Dense(self.action_size))
        model.compile(loss=self._huber_loss, optimizer=SGD(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())
    
    def soft_update(self, w):
        temp = []
        for i in range(len(self.model.get_weights())):
            temp.append(w * self.model.get_weights()[i] + (1 - w) * self.target_model.get_weights()[i])
        self.target_model.set_weights(temp)
    
    def remember(self, state, action, reward, next_state, done):  # store_transition
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-x))
        return s

    def fast_online_act(self, state):
        o1 = self.relu(np.dot(state, self.w1) + self.b1)
        o2 = np.dot(o1, self.w2) + self.b2
        return np.argmax(o2[0])

    def online_act(self, state):
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])
    
    def replay(self, episode, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        targets = []
        states = []
        targets_value = 0
        predict_value = 0
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            predict_value += target[0][action]
            
            target[0][action] = reward + (1 - done) * self.gamma * np.amax(self.target_model.predict(next_state)[0])
            targets_value += target[0][action]
            targets.append(target)
            states.append(state)

        states = np.vstack(states)
        targets = np.vstack(targets)
        aver_target_value = targets_value / batch_size
        aver_predict_value = predict_value / batch_size

        history = self.model.fit(states, targets, epochs=1, verbose=0, shuffle=True)
        loss = history.history['loss'][0]

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def load(self, name):
        self.model.load_weights(name)
        self.w1 = self.model.get_weights()[0]
        self.b1 = self.model.get_weights()[1]
        self.w2 = self.model.get_weights()[2]
        self.b2 = self.model.get_weights()[3]

    def save(self, name):
        self.model.save_weights(name)