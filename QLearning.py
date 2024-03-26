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
        self.Q_table = np.zeros((self.N,self.S*self.S,self.A))


    def epsgreedy_action_selection(self, state):

        actions = np.zeros((self.N))

        for i in range(self.N):

            if np.random.random() < self.epsilon:
                action = np.random.choice(self.A)
            else:
                action = argmax(self.Q_table[i, state,:])

            actions[i] = action

        return actions
			
    
    def update(self, state, actions, reward, next_state):
		
        for i in range(self.N):

            G_t = reward + self.gamma*np.max(self.Q_table[i, next_state[i],:])
            self.Q_table = self.Q_table[i, state[i],actions[i]] + self.learning_rate*(G_t-self.Q_table[i, state[i],actions[i]])



