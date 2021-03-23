import torch


import numpy as np 
import torch

class Agent:
    def __init__(self, arms, shape):
        self.model = None
        self.arms = arms
        self.loss_function = None
        self.softmax_tau = 6
        self.input_shape = shape
    def softmax(self, A, tau = 1.12):
        softm = np.exp(A / tau) / np.sum(np.exp(A / tau))
        return softm

    def one_hot_vector(self, N, pos, val=1):
        one_hot_vec = np.zeros(N)
        one_hot_vec[pos] = val
        return one_hot_vec

    def one_hot(self, state, shape):
        one_hot_matrix = self.one_hot_vector(shape[0], state.website)
        for elem in self.one_hot_vector(shape[1], state.country):
            one_hot_matrix = np.append(one_hot_matrix, elem)

        for elem in self.one_hot_vector(shape[2], state.age):
            one_hot_matrix = np.append(one_hot_matrix, elem)
        return one_hot_matrix

    def train_single(self, env, epochs=5000, learning_rate=1e-2):
        # Convert to torch variable
        current_state = torch.Tensor(self.one_hot_vector(self.arms, env.get_state())) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        rewards = []

        for i in range(epochs):
            y_pred = self.model(current_state)
            av_softmax = self.softmax(y_pred.data.numpy(), tau=self.softmax_tau)
            # Making sure it sums to 1 / Why?
            av_softmax = av_softmax/np.sum(av_softmax)
            # Choose new action based on probabilty
            choice = np.random.choice(self.arms, p=av_softmax)
            current_reward = env.choose_arm(choice) 

            # Convert torch Tensor to numpy array
            one_hot_reward = y_pred.data.numpy().copy()
            one_hot_reward[choice] = current_reward
            rewards.append(current_reward)

            # Train
            reward = torch.Tensor(one_hot_reward)
            loss = self.loss_function(y_pred, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update State
            current_state = torch.Tensor(self.one_hot_vector(self.arms, env.get_state()))

        return np.array(rewards)

    def train_multiple(self, env, epochs=5000, learning_rate=1e-2):
        # Convert to torch variable
        current_state = torch.Tensor(self.one_hot(env.get_state(), self.input_shape)) 
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        rewards = []

        for i in range(epochs):
            y_pred = self.model(current_state)
            av_softmax = self.softmax(y_pred.data.numpy(), tau=self.softmax_tau)
            # Making sure it sums to 1 / Why?
            av_softmax = av_softmax/np.sum(av_softmax)
            # Choose new action based on probabilty
            choice = np.random.choice(self.arms, p=av_softmax)
            current_reward = env.choose_arm(choice) 

            # Convert torch Tensor to numpy array
            one_hot_reward = y_pred.data.numpy().copy()
            one_hot_reward[choice] = current_reward
            rewards.append(current_reward)

            # Train
            reward = torch.Tensor(one_hot_reward)
            loss = self.loss_function(y_pred, reward)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update State
            current_state = torch.Tensor(self.one_hot(env.get_state(), self.input_shape))

        return np.array(rewards)