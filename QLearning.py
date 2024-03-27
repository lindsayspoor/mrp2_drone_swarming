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
        self.obs_angles = -np.linspace(-np.pi, np.pi, self.S)

        #initialize Q-table
        self.Q_table = np.zeros((self.N,self.S,self.A))


    def epsgreedy_action_selection(self, observations):

        actions = np.zeros((self.N))

        for i in range(self.N):

            if np.random.random() < self.epsilon:
                action = np.random.choice(self.A)
            else:
                obs = observations[i]
                obs_index = np.argwhere(self.obs_angles == obs)[0][0]
                action = argmax(self.Q_table[i, obs_index,:])


            actions[i] = action

        return actions
			
    
    def update(self, observations, actions, reward, next_observations):
		
        for i in range(self.N):
            obs_index = np.argwhere(self.obs_angles == observations[i])[0][0]
            next_obs_index = np.argwhere(self.obs_angles == next_observations[i])[0][0]

            # reward = -abs(observations[i])

            G_t = reward + self.gamma*np.max(self.Q_table[i, next_obs_index,:])
            Q_value = self.Q_table[i, obs_index,int(actions[i])] + self.learning_rate*(G_t-self.Q_table[i, obs_index,int(actions[i])])
            # print(f"{Q_value=}")
            self.Q_table[i,obs_index, int(actions[i])] = self.Q_table[i, obs_index,int(actions[i])] + self.learning_rate*(G_t-self.Q_table[i, obs_index,int(actions[i])])



