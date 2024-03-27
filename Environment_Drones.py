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
        self.max_steps = settings["max_steps"]
        # self.action_angles = -np.linspace(-np.pi, np.pi, self.k_a)
         
        self.action_angles = -np.linspace(-np.pi/3, np.pi/3, self.k_a)
        self.obs_angles = -np.linspace(-np.pi, np.pi, self.k_s)
        self.done = False
        self.counter = 0
        self.alignment_costs = []

        #self.state = self.reset()
        self.actions = self.action_angles

    def initialize_positions(self):
        '''Initialize the positions of all N drones randomly.'''

        positions = np.zeros((self.N, self.dim))
        for i in range(self.N):
            for j in range(self.dim):
                positions[i,j] = np.random.uniform(-1,1)*0.5*self.L # centre of box is (0,0), box has x and y limitis of -L/2, L/2
        

        return positions



    def initialize_directions(self):
        '''Initialize the flight directions of all N drones.'''

        directions = np.random.choice(self.obs_angles, size=self.N)
        # directions = [self.obs_angles[0] for i in range(self.N)]

        return directions
    


    def compute_velocities(self, direction_angles):
        '''Compute velocity vector given the direction angles of all N drones.'''

        velocities = np.zeros((self.N, self.dim))
        if self.dim == 2:
            velocities[:,0] = np.cos(direction_angles)
            velocities[:,1] = np.sin(direction_angles)

        #print(f"{np.linalg.norm(velocities,axis=1)}")

        return velocities


    def compute_observations(self, observations, velocities):
        '''Computes the observations for all drones, which is the angle between drone i w.r.t. all other drones for all N drones.
        This angle is discretized over k_s observational angles.'''


        for i in range(self.N):
            v_j = np.zeros((self.N, self.dim))
            for j in range(self.N):
                if j!=i:
                    v_j[j,:] = velocities[j,:]

            sum_v_j = np.sum(v_j, axis=0)

            norm_sum_v_j = np.linalg.norm(sum_v_j)
            observation = np.arccos((np.dot(sum_v_j, velocities[i,:])) / norm_sum_v_j)

            
            selected_obs_angle = self.find_nearest(self.obs_angles, observation)


            observations[i] = selected_obs_angle

        self.observations = observations
        
        return self.observations


        

    def reset(self):
        '''Reset the environment to initial setup.'''


        self.done = False
        self.counter = 0

        # initialize positions and flight directions of all N drones
        self.positions = self.initialize_positions()
        self.directions = self.initialize_directions()
        self.velocities = self.compute_velocities(self.directions)


        self.state = np.zeros((self.N, self.dim + 1))
        self.observations = np.zeros((self.N))
        self.state = self.update_state()
        self.observations = self.compute_observations(self.observations, self.velocities)

        
        return self.state, self.observations


    def update_state(self):
        '''Update the state given the current positions and directions of the drones.'''

        self.state[:,0:(self.dim)] = self.positions
        self.state[:,self.dim] = self.directions

        return self.state



    def update_positions(self):
        '''Update the positions of all drones.'''

        self.positions += self.v0*self.velocities

        return self.positions


    def update_velocities(self, i, angle):
        '''Update the velocities of the i-th drone given the rotation angle.'''



        if self.dim==2:
            self.velocities[i,0] = (self.velocities[i,0]*np.cos(angle) - self.velocities[i,1]*np.sin(angle))
            self.velocities[i,1] = self.velocities[i,0]*np.sin(angle) + self.velocities[i,1]*np.cos(angle)

        self.velocities[i,:] = self.velocities[i,:] / np.linalg.norm(self.velocities[i,:])


        return self.velocities


    def find_nearest(self, array, value):
        '''Finds element of array closest to given value.'''

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return array[idx]
    

    def update_directions(self, i):
        '''Update the directions of the i-th drone according to its velocity.'''

        new_angle = self.find_nearest(self.obs_angles,np.arccos(self.velocities[i,0]))

        self.directions[i] = new_angle

        return self.directions


    def alignment_cost(self):
        '''The alignment of the drones is defined by the order parameter of the system.'''
        
        phi_t = (1/self.N)*np.linalg.norm(np.sum(self.velocities, axis=0), axis=0)

        self.alignment_costs.append(phi_t)
        return phi_t
    

    def compactness_cost(self):

        raise NotImplementedError


    def cost_function(self):

        total_cost = -self.alignment_cost() #-self.compactness_cost()

        return total_cost



    def step(self, actions):
        '''For the i-th drone, action is the k-th element from the action space, i.e. the k-th component of the circle.'''

        for i in range(self.N):

            angle = self.action_angles[int(actions[i])]


            self.velocities = self.update_velocities(i, angle)
            self.directions = self.update_directions(i)


        self.positions = self.update_positions()



        next_state = self.update_state()
        next_observations = self.compute_observations(self.observations, self.velocities)


        # the system gets a collective reward because they have a collective goal
        reward = (-self.cost_function())*2-1


        self.counter += 1

        if self.counter == self.max_steps:
            self.done =  True
            return next_state, next_observations, reward, self.done
        
        else:
            self.done = False
            return next_state, next_observations, reward, self.done




    def render(self):

        raise NotImplementedError