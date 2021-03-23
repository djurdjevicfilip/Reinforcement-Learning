from Gridworld import Gridworld
import torch
import numpy as np
import random 
import matplotlib.pyplot as plt


### GridWorld with Catastrophic Forgetting
### Trained only on the static version of the game, which means that the positions of the objects do not change

game = Gridworld(size=4, mode='static')

l1 = 64 # input
l2 = 150
l3 = 100
l4 = 4 # output

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.ReLU(),
    torch.nn.Linear(l2, l3),
    torch.nn.ReLU(),
    torch.nn.Linear(l3, l4)
)

loss_fn = torch.nn.MSELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

def get_state(game):
    # Adding noise, since most of the input are zeros | can also help with overfitting
    state = game.board.render_np().reshape(1, 64) + np.random.rand(1, 64)/10.0
     # To torch tensor
    return torch.from_numpy(state).float()


def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(mode=mode)
    state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
    state = torch.from_numpy(state_).float()
    if display:
        print("Initial State:")
        print(test_game.display())
    status = 1
    while(status == 1): #A
        qval = model(state)
        qval_ = qval.data.numpy()
        action_ = np.argmax(qval_) #B
        action = action_set[action_]
        if display:
            print('Move #: %s; Taking action: %s' % (i, action))
        test_game.makeMove(action)
        state_ = test_game.board.render_np().reshape(1,64) + np.random.rand(1,64)/10.0
        state = torch.from_numpy(state_).float()
        if display:
            print(test_game.display())
        reward = test_game.reward()
        if reward != -1:
            if reward > 0:
                status = 2
                if display:
                    print("Game won! Reward: %s" % (reward,))
            else:
                status = 0
                if display:
                    print("Game LOST. Reward: %s" % (reward,))
        i += 1
        if (i > 15):
            if display:
                print("Game lost; too many moves.")
            break
    
    win = True if status == 2 else False
    return win

action_set = {
 0: 'u',
 1: 'd',
 2: 'l',
 3: 'r',
}

gamma = 0.9
epsilon = 1.0

epochs = 400
losses = []
for i in range(epochs):
    # Start new game
    game = Gridworld(size = 4, mode = 'static')

    state1 = get_state(game)

    status = 1
    print(i)
    while status:
        Q_val_ = model(state1)
        Q_val = Q_val_.data.numpy()

        if(random.random() < epsilon):
            choice = np.random.randint(0, 4) # Exploration
        else:
            choice = np.argmax(Q_val) # Exploitation

        action = action_set[choice]

        game.makeMove(action)

        state2 = get_state(game)
        reward = game.reward()

        with torch.no_grad(): # Since we won't backpropagate ?
            newQ = model(state2)

        maxQ = torch.max(newQ)

        if reward == -1:
            Y = reward + (gamma*maxQ)
        else:
            Y = np.float(reward)
        
        # Creates a copy...
        X = Q_val_.squeeze()[choice]
        
        Y = torch.Tensor([Y]).detach()
        
        loss = loss_fn(X, Y)
        optimizer.zero_grad()
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        state1 = state2

        # End game
        if reward != -1:
            status = 0
    
    if epsilon > 0.1:
        epsilon -= 1/epochs
            
test_model(model, 'static')


plt.figure(figsize=(10,7))
plt.plot(losses)
plt.xlabel("Iterations",fontsize=22)
plt.ylabel("Loss",fontsize=22)


plt.show()



