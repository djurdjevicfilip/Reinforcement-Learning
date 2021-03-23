import numpy as np
import matplotlib.pyplot as plt
import torch
from ContextBandit import ContextBanditSingleElement
from Agent import Agent

### AD PLACEMENT w/ a single element state ###
# State { current_website }
# Action -> Placing an ad
# Agent -> Neural network

def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


arms = 100
N, D_in, H, D_out = 1, arms, 500, arms



# Creating the agent
agent = Agent(arms)

agent.model = torch.nn.Sequential(
 torch.nn.Linear(D_in, H),
 torch.nn.ReLU(),
 torch.nn.Linear(H, D_out),
 torch.nn.ReLU(),
)

agent.loss_function = torch.nn.MSELoss()

# Adjust tau
agent.softmax_tau = 6

env = ContextBanditSingleElement(arms)

rewards = agent.train(env)
print(rewards)
plt.plot(running_mean(rewards,N=500))
plt.show()