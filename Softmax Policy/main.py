# Choosing cancer treatments 

import numpy as np
import matplotlib.pyplot as plt
import random

# RECORD STRUCTURE: {number of values observed; average;}

# Returns an array of probabilities
def softmax(A, tau = 1.12):
    softm = np.exp(A / tau) / np.sum(np.exp(A / tau))
    return softm

# Returns the reward
def get_reward(prob, n=10):
    reward = 0
    for i in range(n):
        if random.random() < prob:
            reward += 1
    return reward

# Updates the record
def update_record(record, action, reward):
    new_reward = (record[action, 0] * record[action, 1] + reward) / (record[action, 0] + 1)
    record[action, 0] += 1
    record[action, 1] = new_reward
    return record

# Initial parameters
n = 10
probs = np.random.rand(n) # Random probabilities
record = np.zeros((n, 2))

fig, ax = plt.subplots(1, 1)
ax.set_xlabel("Plays")
ax.set_ylabel("Avg Reward")
fig.set_size_inches(9,5)

rewards = [0]
for i in range(500):
    p = softmax(record[:, 1])
    choice = np.random.choice(np.arange(n), p=p)
    reward = get_reward(probs[choice])
    record = update_record(record, choice, reward)

    mean_reward = ((i+1) * rewards[-1] + reward) / (i+2)
    rewards.append(mean_reward)

ax.scatter(np.arange(len(rewards)), rewards)

plt.show()