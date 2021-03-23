import numpy as np
import matplotlib.pyplot as plt
import torch
from ContextBandit import ContextBandit
from Agent import Agent

### AD PLACEMENT w/ a multiple element state ###
# State { current_website, user_country, user_age }
# Action -> Placing an ad
# Agent -> Neural network

def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


actions = 10
websites = ages = countries = 10
N, D_in, H, D_out = 1, ages+websites+countries, 500, actions

# Creating the agent
agent = Agent(actions, shape = (websites, countries, ages))

agent.model = torch.nn.Sequential(
 torch.nn.Linear(D_in, H),
 torch.nn.ReLU(),
 torch.nn.Linear(H, D_out),
 torch.nn.ReLU(),
)

agent.loss_function = torch.nn.MSELoss()

# Adjust tau
agent.softmax_tau = 0.6
env = ContextBandit(actions, websites, countries, ages)

rewards = agent.train_multiple(env, epochs = 50000, learning_rate = 1e-2)

plt.plot(running_mean(rewards,N=5000))
plt.show()