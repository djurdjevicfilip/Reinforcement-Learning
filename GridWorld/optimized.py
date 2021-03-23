from Gridworld import Gridworld
import torch
import numpy as np
import random 
import matplotlib.pyplot as plt
from collections import deque
import copy
### GridWorld with Experience Replay(no catas. forgetting) and satabilization with a target network
### Trained only on the static version of the game, which means that the positions of the objects do not change
if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
print(dev)
device = torch.device(dev)
game = Gridworld(size=5, mode='static')

input_size = 100

l1 = input_size # input
l2 = 300
l3 = 200
l4 = 80
l5 = 4 # output


def get_state(game):
    # Adding noise, since most of the input are zeros | can also help with overfitting
    state = game.board.render_np().reshape(1, input_size) + np.random.rand(1, input_size)/10.0
     # To torch tensor
    return torch.from_numpy(state).float()


def test_model(model, mode='static', display=True):
    i = 0
    test_game = Gridworld(size=5, mode=mode)
    state_ = test_game.board.render_np().reshape(1,input_size) + np.random.rand(1,input_size)/10.0
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
        state_ = test_game.board.render_np().reshape(1,input_size) + np.random.rand(1,input_size)/10.0
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

def experience_replay(replay, batch_size, gamma, model, model2):
    minibatch = random.sample(replay, batch_size)
    state1_batch = torch.cat([s1 for (s1, a, r, s2, d) in minibatch])
    action_batch = torch.Tensor([a for (s1,a,r,s2,d) in minibatch])
    reward_batch = torch.Tensor([r for (s1,a,r,s2,d) in minibatch])
    state2_batch = torch.cat([s2 for (s1,a,r,s2,d) in minibatch])
    done_batch = torch.Tensor([d for (s1,a,r,s2,d) in minibatch])

    Q1 = model(state1_batch) 
    with torch.no_grad():
        Q2 = model2(state2_batch) 
    Y = reward_batch + gamma * ((1 - done_batch) * torch.max(Q2,dim=1)[0])
    # Select only the elements that were chosen / actions
    X = Q1.gather(dim=1,index=action_batch.long().unsqueeze(dim=1)).squeeze()
    return X, Y

action_set = {
 0: 'u',
 1: 'd',
 2: 'l',
 3: 'r',
}

epochs = 100
sync_freq = 500 # Target update frequency
def train(mem_size, batch_size, sync_freq, epochs=500, print_epoch = False):
    
    model = torch.nn.Sequential(
        torch.nn.Linear(l1, l2),
        torch.nn.ReLU(),
        torch.nn.Linear(l2, l3),
        torch.nn.ReLU(),
        torch.nn.Linear(l3, l4),
        torch.nn.ReLU(),
        torch.nn.Linear(l4, l5)
    )
    model2 = copy.deepcopy(model) 
    model2.load_state_dict(model.state_dict()) 
    loss_fn = torch.nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    replay = deque(maxlen=mem_size) 
    gamma = 0.9
    epsilon = 1.0
    j = 0
    losses = []
    for i in range(epochs):
            # Start new game
        game = Gridworld(size = 5, mode = 'random')

        state1 = get_state(game)

        status = 1
        if print_epoch:
            print(i)
        while status:
            j+=1
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
            done = True if reward > 0 else False
            exp = (state1, choice, reward, state2, done) 
            replay.append(exp)

            if(len(replay)>batch_size):
            
                X, Y = experience_replay(replay, batch_size, gamma, model, model2)
                loss = loss_fn(X, Y.detach())
                optimizer.zero_grad()
                loss.backward()
                losses.append(loss.item())
                optimizer.step()
                # Target update
                if j % sync_freq == 0: 
                    model2.load_state_dict(model.state_dict())
            state1 = state2

            # End game
            if reward != -1:
                status = 0
        
        if epsilon > 0.1:
            epsilon -= 1/epochs

    max_games = 1000
    wins = 0
    for i in range(max_games):
        win = test_model(model, mode='random', display=False)
        if win:
            wins += 1
    win_perc = float(wins) / float(max_games)
    print("Games played: {0}, # of wins: {1}".format(max_games,wins))
    print("Win percentage: {}".format(win_perc))

    if print_epoch:
        plt.figure(figsize=(10,7))
        plt.plot(losses)
        plt.xlabel("Iterations",fontsize=22)
        plt.ylabel("Loss",fontsize=22)


        plt.show()
    return win_perc

def choose_hyperparameters(mem_sizes, batch_sizes, sync_freqs, epochs=500):
    acc = 0
    (mem, batch, sync) = 0,0,0
    i = 0
    for mem_size in mem_sizes:
        for batch_size in batch_sizes:
            for sync_freq in sync_freqs:
                
                i+=1
                print("Hyper "+str(i))
                print((mem_size, batch_size, sync_freq))
                new_acc = train(mem_size, batch_size, sync_freq, epochs)
                if new_acc > acc:
                    acc = new_acc
                    (mem, batch, sync) = mem_size, batch_size, sync_freq
                    
    print("MAX: "+str(acc))
    print((mem, batch, sync))



mem_sizes = [500, 1000, 2000]
batch_sizes = [20, 50, 200]
sync_freqs = [10, 200, 500]

# choose_hyperparameters(mem_sizes, batch_sizes, sync_freqs, epochs = 500)

mem_size, batch_size, sync_freq = (1000, 20, 10)
train(mem_size, batch_size, sync_freq, epochs=15000, print_epoch=True)


