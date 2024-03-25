import numpy as np
from functions import argmax, softmax


class QLearning:
	
    def __init__(self, settings):


        self.S = settings["S"]
        self.A = settings["A"]
        self.learning_rate = settings["learning_rate"]
        self.gamma = settings["gamma"]
        self.epsilon = settings["epsilon"]
        self.N = settings["N"]

        #initialize Q-table
        self.Q_table = np.zeros((self.N,self.S,self.A))


    def epsgreedy_action_selection(self, state, i):

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.A)
        else:
            action = argmax(self.Q_table[i, state,:])
        
        return action
			
    
    def update(self, i, state, action, reward, next_state):
		
        G_t = reward + self.gamma*np.max(self.Q_table[i, next_state,:])
        self.Q_table = self.Q_table[i, state,action] + self.learning_rate*(G_t-self.Q_table[i, state,action])



