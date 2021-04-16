import gym
import torch
import numpy as np
from matplotlib import pyplot as plt

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y
env = gym.make('CartPole-v0')

l1 = 4 
l2 = 150 
l3 = 2 

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax()
)

def discount_rewards(rewards, gamma = 0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma,torch.arange(lenr).float()) * rewards 
    disc_return /= disc_return.max() 
    return disc_return

def loss_fn(preds, r): 
    return -1 * torch.sum(r * torch.log(preds))

MAX_DURATION = 200
MAX_EPISODES = 500
learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
score = []
for episode in range(MAX_EPISODES): 
    curr_state = env.reset()
    done = False
    transitions = [] 
    print(episode)

    for t in range(MAX_DURATION):
        action_probability = model(torch.from_numpy(curr_state).float())
        action = np.random.choice(np.array([0,1]),p=action_probability.data.numpy())
        prev_state = curr_state

        curr_state, _, done, info = env.step(action)
        transitions.append((prev_state, action, t+1)) 
        if done:
            break

    # Our reward
    episode_len = len(transitions)

    score.append(episode_len)

    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) # Flip the rewards
    state_batch = torch.Tensor([s for (s,a,r) in transitions])
    action_batch = torch.Tensor([a for (s,a,r) in transitions])

    discounted_rewards = discount_rewards(reward_batch)

    prediction_batch = model(state_batch)

    # select by action
    prob_batch = prediction_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
    
    loss = loss_fn(prob_batch, discounted_rewards)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

score = np.array(score)
avg_score = running_mean(score, 50)

plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_score, color='green')

plt.show()