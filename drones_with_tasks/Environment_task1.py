import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt



class Env_Task1(gym.Env):
    """Environment for drones to perform task 1, which is go from area A to area B as a collective swarm of N drones."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, settings):
        super().__init__()

        self.N = settings["N"] # N is the numer of drones the swarm consists of
        self.k_a = settings["k_a"] # k_a equally spaced angles for the actions in range [-theta_max, theta_max]
        self.k_s = settings["k_s"] # k_s equally spaced angles for the direction angle in the range [-pi, pi)
        self.theta_max = settings["theta_max"]
        self.L = settings["L"] # size of grid of total enviroment (LxL)
        self.La_x = settings["La_x"] # x-size of area A
        self.La_y = settings["La_y"] # y-size of area A
        self.Lb_x = settings["Lb_x"] # x-size of area B
        self.Lb_y = settings["Lb_y"] # y-size of area B
        self.origin_Ax = settings["origin_Ax"] # x origin of area A
        self.origin_Ay = settings["origin_Ay"] # y origin of area A
        self.origin_Bx = settings["origin_Bx"] # x origin of area B
        self.origin_By = settings["origin_By"] # y origin of area B
        self.max_timesteps = settings["max_timesteps"] # maximum amount of timesteps to play game before truncation
        self.step_reward = settings["step_reward"]
        self.goal_reward = settings["goal_reward"]

        self.action_angles = np.linspace(-self.theta_max, self.theta_max, self.k_a)
        self.direction_angles =-np.linspace(-np.pi, np.pi, self.k_s)
        self.counter = 0 # updates each step to count what timestep we are in
        self.done = False
        self.truncated = False
        self.collective_reward = 0

        #self.action_space = spaces.MultiBinary([self.N, self.k_a]) # for each agent there are k_a possible actions to choose from, selects for all N agents the k'th index for the action angles
        self.action_space = spaces.Discrete(self.N*self.k_a) # for each agent there are k_a possible actions to choose from, selects for all N agents the k'th index for the action angles
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.MultiBinary([self.L, self.L]) # each cell can occupy at the most one drone, 0 =  no drone, 1 = drone occupation

        # initialize grid
        #self.grid = np.zeros((self.L, self.L))
        self.initialize_grid()
        
        self.drone_grid_positions = np.zeros((self.L, self.L))
        self.drone_positions_memory = []
        self.initialize_drones()


        # construct rewarding of grid
        self.reward_grid = np.zeros((self.L,self.L))
        self.initialize_rewards()

    def initialize_grid(self):

        self.grid_positions = [[i,j] for i in range(self.L) for j in range(self.L)]
        self.grid_A_positions = [[i,j] for i in range(self.origin_Ax, self.origin_Ax+self.La_x) for j in range(self.origin_Ay, self.origin_Ay+self.La_y)]
        self.grid_B_positions = [[i,j] for i in range(self.origin_Bx, self.origin_Bx+self.Lb_x) for j in range(self.origin_By, self.origin_By+self.Lb_y)]


    def initialize_drones(self):
        '''Initialize drone positions within area A'''

        # for now only work with 2 drones to see if the environment works
        # choose random initial locations

        self.drone_grid_indices = np.random.choice(np.arange(self.La_x*self.La_y), size=self.N, replace=False)

        for i in range(self.N):
            self.drone_grid_positions[self.grid_A_positions[self.drone_grid_indices[i]][0],self.grid_A_positions[self.drone_grid_indices[i]][1]] = 1
        
            self.drone_positions_memory.append([self.grid_A_positions[self.drone_grid_indices[i]][0],self.grid_A_positions[self.drone_grid_indices[i]][1]])

        self.drone_directions = np.random.choice(self.direction_angles, size=self.N)
        self.drone_velocities = self.compute_velocities(self.drone_directions)
        

    def initialize_rewards(self):

        self.reward_grid[:,:] = self.step_reward
        self.reward_grid[self.grid_B_positions[0][0]:(self.grid_B_positions[-1][0]+1), self.grid_B_positions[0][1]:(self.grid_B_positions[-1][1]+1)] = self.goal_reward


    def compute_velocities(self, direction_angles):
        '''Compute velocity vector given the direction angles of all N drones.'''

        velocities = np.zeros((self.N, 2))

        velocities[:,0] = np.cos(direction_angles)
        velocities[:,1] = np.sin(direction_angles)


        return velocities

    def find_nearest(self, array, value):
        '''Finds element of array closest to given value.'''

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return array[idx]

    def update_drone_directions(self, i, directions, velocities):
        '''Update the directions of the drones according to their velocities.'''


        new_angle = self.find_nearest(self.direction_angles,np.arccos(velocities[i,0]))

        directions[i] = new_angle

        return directions

    def update_drone_positions(self, i, positions, velocities):
        '''Update the positions of all drones.'''


        new_drone_grid_values = self.grid_positions[self.drone_grid_indices[i]] + velocities[i,:]

        new_drone_grid_index_x = self.find_nearest(np.arange(self.L), new_drone_grid_values[0])
        new_drone_grid_index_y = self.find_nearest(np.arange(self.L), new_drone_grid_values[1])

        positions[:,:] = 0
        positions[new_drone_grid_index_x,new_drone_grid_index_y] = 1


        return positions




    def compute_angles(self, i, actions):
        '''Compute turning angles from given actions.'''

        new_angles = np.zeros((self.N))

        action_index = np.argwhere(actions[i] == 1)[0][0]
        new_angles[i] = self.action_angles[action_index]
        
        return new_angles


    def update_drone_velocities(self, i, velocities, angles):
        '''Update the velocities of the i-th drone given the rotation angle.'''


        velocities[i,0] = velocities[i,0]*np.cos(angles[i]) - velocities[i,1]*np.sin(angles[i])
        velocities[i,1] = velocities[i,0]*np.sin(angles[i]) + velocities[i,1]*np.cos(angles[i])

        # normalize the velocities
        velocities[i,:] = velocities[i,:] / np.linalg.norm(velocities[i,:])


        return velocities

    def step(self, actions):

        actions = actions.reshape((self.N, self.k_a))

        self.collective_reward = 0

        # compute turning angles from given actions
        if self.counter == self.max_timesteps:
            # self.collective_reward = 0
            self.counter = 0
            self.truncated = True
            return self.drone_grid_positions, self.collective_reward, self.done, self.truncated, {"max steps reached"}

        

        for i in range(self.N):
            new_angles = self.compute_angles(i, actions)

            self.drone_velocities = self.update_drone_velocities(i, self.drone_velocities, new_angles)

            self.drone_directions = self.update_drone_directions(i, self.drone_directions, self.drone_velocities)

            self.drone_grid_positions = self.update_drone_positions(i, self.drone_grid_positions, self.drone_velocities)


            reward_index = np.argwhere(self.drone_grid_positions==1)[0]
            reward_drone_i = self.reward_grid[reward_index[0],reward_index[1]]

            self.collective_reward += reward_drone_i

        self.counter += 1

        return self.drone_grid_positions, self.collective_reward, self.done, self.truncated, {"continue"}
        #return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        
        self.collective_reward = 0
        self.done = False
        self.truncated = False
        self.counter = 0

        self.initialize_grid()
        
        self.drone_grid_positions = np.zeros((self.L, self.L))
        self.initialize_drones()


        return self.drone_grid_positions, None

    def draw_image(self):
        fig, ax = plt.subplots(figsize = (10,10))
        a=1/(self.L)


        patch_A = plt.Polygon([[a*(self.origin_Ax), a*(self.origin_Ay)], [a*(self.origin_Ax+La_x), a*(self.origin_Ay)], [a*(self.origin_Ax+La_x), a*(self.origin_Ay+La_y)], [a*(self.origin_Ax), a*(self.origin_Ay+La_y)] ], fc = 'lightblue')
        ax.add_patch(patch_A)

        patch_B = plt.Polygon([[a*(self.origin_Bx), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By+Lb_y)], [a*(self.origin_Bx), a*(self.origin_By+Lb_y)] ], fc = 'lightgreen')
        ax.add_patch(patch_B)


        # Draw grid
        for x in range(self.L):
            for y in range(self.L):
                pos=(a*x, a*y)
                width=a
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black', linewidth=0.2 )
                ax.add_patch(lattice)

                if self.drone_grid_positions[x,y] == 1:
                    patch_drone = plt.Rectangle(pos, width, width, fc = 'darkblue', zorder = 10)
                    ax.add_patch(patch_drone)




        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')

    def render(self):
        
        fig, ax = plt.subplots(figsize = (10,10))
        a=1/(self.L)


        patch_A = plt.Polygon([[a*(self.origin_Ax), a*(self.origin_Ay)], [a*(self.origin_Ax+La_x), a*(self.origin_Ay)], [a*(self.origin_Ax+La_x), a*(self.origin_Ay+La_y)], [a*(self.origin_Ax), a*(self.origin_Ay+La_y)] ], fc = 'lightblue')
        ax.add_patch(patch_A)

        patch_B = plt.Polygon([[a*(self.origin_Bx), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By+Lb_y)], [a*(self.origin_Bx), a*(self.origin_By+Lb_y)] ], fc = 'lightgreen')
        ax.add_patch(patch_B)


        # Draw grid
        for x in range(self.L):
            for y in range(self.L):
                pos=(a*x, a*y)
                width=a
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black', linewidth=0.2 )
                ax.add_patch(lattice)

                if self.drone_grid_positions[x,y] == 1:
                    patch_drone = plt.Rectangle(pos, width, width, fc = 'darkblue', zorder = 10)
                    ax.add_patch(patch_drone)




        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

        #plt.pause(0.1)





    def close(self):
        self.drone_grid_positions = None

if __name__ == "__main__":

    N = 2
    k_a = 6
    k_s = 10
    theta_max = np.pi / 6
    L = 100
    La_x = 20
    La_y = 20
    Lb_x = 20
    Lb_y = 20
    origin_Ax = 0
    origin_Ay = 40
    origin_Bx = L-Lb_x
    origin_By = 40
    max_timesteps = 100
    step_reward = -1
    goal_reward = 300

    n_timesteps = 100

    settings = {"N": N,
                "k_a": k_a,
                "k_s": k_s,
                "theta_max": theta_max,
                "L": L,
                "La_x": La_x,
                "La_y": La_y,
                "Lb_x": Lb_x,
                "Lb_y": Lb_y,
                "origin_Ax": origin_Ax,
                "origin_Ay": origin_Ay,
                "origin_Bx": origin_Bx,
                "origin_By": origin_By,
                "max_timesteps": max_timesteps,
                "step_reward": step_reward,
                "goal_reward": goal_reward
                }
    
    env = Env_Task1(settings=settings)
    # print(env.grid_positions)


    #env.reset()

    env.render()
    
    '''
    for i in range(n_timesteps):

        actions = np.array([[0,1,0,0,0,0],[0,0,1,0,0,0]])
        env.step(actions)

        
        if (i % 30) == 0:

            env.render()
        
    #env.render()
    '''