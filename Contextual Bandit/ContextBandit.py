import numpy as np
import random

# Single element state, same number of states and actions ***
class ContextBanditSingleElement:
    def __init__(self, arms = 10):
        self.arms = arms
        self.init_distribution(arms)
        self.update_state()

    # Number of states equals number of arms for simplicity
    def init_distribution(self, arms):
        self.bandit_matrix = np.random.rand(arms,arms)

    def reward(self, prob):
        reward = 0
        for i in range(self.arms):
            if random.random() < prob:
                reward += 1
        return reward
    
    def get_state(self):
        return self.state

    def update_state(self):
        self.state = np.random.randint(0,self.arms)
    
    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])
        
    def choose_arm(self, arm): 
        reward = self.get_reward(arm)
        self.update_state()
        return reward

# Multiple element state, N websites, M countries, P ages (yup, limited age)


class State:
    def __init__(self, website, country, age):
        self.website = website
        self.country = country
        self.age = age

class ContextBandit:
    def __init__(self, actions, websites, countries, ages):
        self.websites = websites
        self.countries = countries
        self.ages = ages
        self.actions = actions
        self.init_distribution(self.websites, self.countries, self.ages, self.actions)
        self.update_state()

    # Number of states equals number of arms for simplicity
    def init_distribution(self, websites, countries, ages, actions):
        self.bandit_matrix = np.random.rand(websites,countries,ages,actions)
        
    def reward(self, prob):
        reward = 0
        for i in range(self.actions):
            if random.random() < prob:
                reward += 1
        return reward
    
    def get_state(self):
        return self.state

    def update_state(self):
        self.state = State(np.random.randint(0,self.websites),
                           np.random.randint(0,self.countries),
                           np.random.randint(0,self.ages))
    
    def get_reward(self, arm):
        current_state = self.get_state()
        return self.reward(self.bandit_matrix[current_state.website][current_state.country][current_state.age][arm])
        
    def choose_arm(self, arm): 
        reward = self.get_reward(arm)
        self.update_state()
        return reward
