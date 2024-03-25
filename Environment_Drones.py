import numpy as np
import matplotlib.pyplot as plt

class EnvironmentDrones:

    def __init__(self, settings):

        self.dim = settings["dim"]
        self.N =  settings["N"]
        self.L = settings["L"]
        self.periodic = settings["periodic"]
        self.v0 = settings["v0"]
        self.R = settings["R"]
        self.k = settings["k"]
        self.angles = np.linspace(-np.pi, np.pi, self.k)

        self.done = False

        self.s0 = self.reset()

    def reset(self):
        '''Reset the environment to initial setup.'''

        # initialize state for N agents, with all random x,y (or z if dim=3) positions +1 for direction angle, N-1 agents w.r.t. the i-th agent
        state = np.zeros((self.N, self.dim+1, self.N-1))
        for i in range(self.N):
            for d in range(self.dim):
                for j in range(self.N-1):
                    state[i,d,j] = np.random.random()*self.L

        #return state
        raise NotImplementedError

    def alignment_cost(self):

        raise NotImplementedError
    

    def compactness_cost(self):

        raise NotImplementedError


    def cost_function(self):

        raise NotImplementedError


    def step(self):
        '''Update the environment by one timestep and return the rewards according to the new state.'''

        #return next_state,reward,done
        raise NotImplementedError
    

    def render(self):

        raise NotImplementedError