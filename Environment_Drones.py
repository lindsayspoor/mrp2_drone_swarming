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
        self.k_s = settings["k_s"]
        self.k_a = settings["k_a"]
        self.angles = np.linspace(-np.pi, np.pi, self.k_a)

        self.done = False

        self.s0 = self.reset()
        self.actions = self.angles

    def initialize_positions(self):
        '''Initialize the positions of all N drones randomly.'''

        positions = np.zeros((self.N, self.dim))
        for i in range(self.N):
            for j in range(self.dim):
                positions[i,j] = np.random.uniform(-1,1)*0.5*self.L # centre of box is (0,0), box has x and y limitis of -L/2, L/2
        

        return positions



    def initialize_directions(self):
        '''Initialize the flight directions of all N drones.'''

        directions = np.random.choice(self.angles, size=self.N)

        return directions
    
    def compute_velocities(self, direction_angles):
        '''Compute velocity vector given the direction angles of all N drones.'''

        velocities = np.zeros((self.N, self.dim))
        if self.dim == 2:
            velocities[:,0] = np.cos(direction_angles)
            velocities[:,1] = np.sin(direction_angles)

        return velocities




    def reset(self):
        '''Reset the environment to initial setup.'''

        '''
        # initialize state for N agents, with all random x,y (or z if dim=3) positions +1 for direction angle, N-1 agents w.r.t. the i-th agent
        state = np.zeros((self.N, self.dim+1, self.N-1))
        for i in range(self.N):
            for j in range(self.N-1):
                for d in range(self.dim):
                    state[i,d,j] = np.random.random()*self.L
                state[i,self.dim,j] = np.random.choice(self.angles)

        return state
        '''

        # initialize positions and flight directions of all N drones
        self.positions = self.initialize_positions()
        self.directions = self.initialize_directions()
        self.velocities = self.compute_velocities(self.directions)


        self.state = np.zeros((self.N, self.dim + 1))
        self.update_state()

        
        return self.state


    def update_state(self):
        '''Update the state given the current positions and directions of the drones.'''
        self.state[:,0:(self.dim)] = self.positions
        self.state[:,self.dim] = self.directions



    def update_positions(self):
        '''Update the positions of all drones.'''

        if self.dim==2:
            self.positions[:,0] = self.positions[:,0] = self.v0 * self.velocities[:,0]#*delta_t, but for now assuming delta_t is 1
        
    def update_velocities(self, i, angle):
        '''Update the velocities of the i-th drone given the rotation angle.'''

        if self.dim==2:
            self.velocities[i,0] = self.velocities[i,0]*np.cos(angle) - self.velocities[i,1]*np.sin(angle)
            self.velocities[i,1] = self.velocities[i,0]*np.sin(angle) + self.velocities[i,1]*np.cos(angle)

    def update_directions(self, i):
        '''Update the directions of the i-th drone according to its velocity.'''

        self.directions[i] = np.arccos(self.velocities[i,0])

    def alignment_cost(self):
        '''The alignment of the drones is defined by the order parameter of the system.'''

        phi_t = (1/self.N)*np.linalg.norm(np.sum(self.velocities))

        raise NotImplementedError
    

    def compactness_cost(self):

        raise NotImplementedError


    def cost_function(self):

        total_cost = self.alignment_cost() #+self.compactness_cost()



    def step(self, actions):
        '''For the i-th drone, action is the k-th element from the action space, i.e. the k-th component of the circle.'''

        for i in range(self.N):

            angle = self.angles[actions[i]]

            self.update_velocities(i, angle)
            self.update_directions(i)

        self.update_positions()

        self.update_state()

        # now, the environment has to assign a reward to each individual drone. This is based on the cost for all drones.






        #return next_state,reward,done
        #raise NotImplementedError
    

    def render(self):

        raise NotImplementedError