import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.patches as patches



class Env_Task1(gym.Env):
    """Environment for drones to perform task 1, which is go from area A to area B as a collective swarm of N drones."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, settings):
        super().__init__()

        self.N = settings["N"] # N is the numer of drones the swarm consists of
        self.k_a = settings["k_a"] # k_a equally spaced angles for the actions in range [-theta_max, theta_max]
        self.k_s = settings["k_s"] # k_s equally spaced angles for the direction angle in the range [-pi, pi)
        self.theta_max = settings["theta_max"]
        self.boundary_width = settings["boundary_width"] # number of grid elements the boundary width consists of
        self.L = settings["L"] # size of grid of total enviroment (LxL)
        self.Rv = settings["Rv"] # visibility Radius for each drone
        #self.l = settings["l"] # size of each grid cell
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
        self.direction_angles = np.linspace(-np.pi+(2*np.pi/self.k_s), np.pi, self.k_s)

        print(f"{self.action_angles=}")
        print(f"{self.direction_angles=}")

        self.counter = 0 # updates each step to count what timestep we are in
        self.done = False
        self.truncated = False
        self.collective_reward = 0

        self.action_space = spaces.Discrete(self.k_a) # for the i-th drone there are k_a possible actions to choose from
        self.observation_space = spaces.MultiBinary([self.N, self.L, self.L, self.k_s]) # observation space is (N, L, L, k_s), for each j-th drone (j!=i and within the visibility range of the i-th drone) the Lx, Ly grid coordinates of all N-1 other agents and the k_s direction angle

        # initialize grid
        self.initialize_grid()
        
        

        self.state = np.zeros((self.N, 3)) # N drones, Lx coordinate, Ly coordinate, k_s angle

        self.initialize_drones()


        # construct rewarding of grid
        self.initialize_rewards()


    def initialize_grid(self):

        self.grid_positions = [[i,j] for i in range(self.L) for j in range(self.L)]
        self.grid_A_positions = [[i,j] for i in range(self.origin_Ax, self.origin_Ax+self.La_x) for j in range(self.origin_Ay, self.origin_Ay+self.La_y)] # define area A on grid
        self.grid_B_positions = [[i,j] for i in range(self.origin_Bx, self.origin_Bx+self.Lb_x) for j in range(self.origin_By, self.origin_By+self.Lb_y)] # define area B on grid



    def initialize_drones(self):
        '''Initialize drone positions within area A.'''

        self.drone_grid_indices = np.random.choice(np.arange(self.La_x*self.La_y), size=self.N, replace=False) # randomly choose initial grid locations for all N drones in area A
        # by initialising the drones on the grid positions and setting replace = False, all drones will never be initialised onto the same grid cell

        self.drone_velocities = np.zeros((self.N,2))
        #self.drone_grid_positions = np.zeros((self.N, 2))
        self.drone_directions = np.random.choice(self.direction_angles, size = self.N) # choose random initial directions for all drones
        # self.drone_directions = [self.direction_angles[7]]
        print(f"{self.drone_directions=}")

        for i in range(self.N):
            # self.drone_grid_positions[self.grid_A_positions[self.drone_grid_indices[i]][0],self.grid_A_positions[self.drone_grid_indices[i]][1]] = 1
            self.state[i, 0:2] =  [self.grid_A_positions[self.drone_grid_indices[i]][0],self.grid_A_positions[self.drone_grid_indices[i]][1]]

            #self.drone_grid_positions[i,:] = self.state[i,0:2]
            self.state[i, 2] = self.drone_directions[i]

            self.drone_velocities[i,:] = self.compute_velocities(self.drone_directions[i]) # compute the initial velocity vecotr for all drones based on the given direction angle
        


    def initialize_rewards(self):
        '''Initialize the rewards per grid cell.'''

        self.reward_grid = np.zeros((self.L,self.L))


        self.reward_grid[:,:] = self.step_reward
        self.reward_grid[self.grid_B_positions[0][0]:(self.grid_B_positions[-1][0]+1), self.grid_B_positions[0][1]:(self.grid_B_positions[-1][1]+1)] = self.goal_reward

        # define boundaries by assigning a very large negative reward to these grid positions
        self.reward_grid[0, :] = -1000
        self.reward_grid[:,0] = -1000
        self.reward_grid[-1,:] = -1000
        self.reward_grid[:,-1] = -1000




    def compute_velocities(self, direction_angle):
        '''Compute velocity vector given the direction angles 1 drone.'''

        velocity = np.zeros((2))

        velocity[0] = np.cos(direction_angle)
        velocity[1] = np.sin(direction_angle)


        return velocity

    def find_nearest(self, array, value):
        '''Finds element of array closest to given value.'''

        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()

        return array[idx]

    def update_drone_directions(self, i, velocities):
        '''Update the directions of the drones according to their velocities.'''

        print(f"the old direction was {self.drone_directions[i]}")
        new_angle = self.find_nearest(self.direction_angles, np.arctan(velocities[i,1]/velocities[i,0]))

        
        self.drone_directions[i] = new_angle

        print(f"the new direction is {self.drone_directions[i]}")

        #return directions


    def update_drone_positions(self, i, velocities):
        '''Update the position of the ith drone.'''


        print(f"the old positions are {self.state[i,0:2]}")
        new_pos = self.state[i,0:2] + velocities[i,:]

        print(f"the new positions are {new_pos}")


        # new_drone_grid_index_x = self.find_nearest(np.arange(self.L), new_pos[0])
        # new_drone_grid_index_y = self.find_nearest(np.arange(self.L), new_pos[1])

        new_pos[0] = new_pos[0] - (new_pos[0] % 1)
        new_pos[1] = new_pos[1] - (new_pos[1] % 1)

        print(f"the new positions are {new_pos}")

        self.state[i,0:2] = new_pos


    # def update_drone_positions(self, i, positions, velocities):
    #     '''Update the positions of all drones.'''

    #     print(self.drone_grid_indices[i])

    #     new_drone_grid_values = self.grid_positions[self.drone_grid_indices[i]] + velocities[i,:]
    #     # new_drone_grid_values = self.state[i,0:2] + velocities[i,:]
    #     print(new_drone_grid_values)

    #     new_drone_grid_index_x = self.find_nearest(np.arange(self.L), new_drone_grid_values[0])
    #     new_drone_grid_index_y = self.find_nearest(np.arange(self.L), new_drone_grid_values[1])

        
    #     self.drone_grid_indices[i] = self.grid_positions.index([new_drone_grid_index_x,new_drone_grid_index_y])
        

    #     positions[:,:] = 0
    #     positions[new_drone_grid_index_x,new_drone_grid_index_y] = 1


    #     return positions




    def compute_angles(self, i, actions):
        '''Compute turning angles from given actions.'''

        new_angles = np.zeros((self.N))

        action_index = np.argwhere(actions[i] == 1)[0][0]
        new_angles[i] = self.action_angles[action_index]

        return new_angles
        
        
    def drone_dispersion_vector(self, i, positions):
        '''Computes the dispersion vector of the i-th drone w.r.t. all other drones.
        positions = drone x,y cartesian coordinates'''

        c = np.zeros((2))
        for j in range(self.N):
            if j != i:
                if np.linalg.norm(positions[i,:] - positions[j,:]) < 1:
                    c -= (positions[i,:] - positions[j,:])

        print(f"the dispersion c={c}")

        return c


    def update_drone_velocities(self, i, angle):
        '''Update the velocities of the i-th drone given the rotation angle.'''

        print(f"the old velocities are {self.drone_velocities[i,:]}")
        print(f"rotating at angle {angle}")

        c = self.drone_dispersion_vector(i, self.state[:,0:2])

        self.drone_velocities[i,0] = self.drone_velocities[i,0]*np.cos(angle) - self.drone_velocities[i,1]*np.sin(angle) + c[0]
        self.drone_velocities[i,1] = self.drone_velocities[i,0]*np.sin(angle) + self.drone_velocities[i,1]*np.cos(angle) + c[1]

        # normalize the velocities
        self.drone_velocities[i,:] = self.drone_velocities[i,:] / np.linalg.norm(self.drone_velocities[i,:])

        print(f"the new velocities are {self.drone_velocities[i,:]}")

    
    def step(self, actions):
        '''Step function for all N actions for all N agents.'''

        obs_N = []
        reward_N = []
        done_N = []
        info_N = {'N':[]}


        for i in range(self.N):
            print(f"updating drone {i}")
            print(f"action is {actions[i]}")

            action_angle = self.action_angles[actions[i]]

            self.update_drone_velocities(i, action_angle)

            self.update_drone_directions(i, self.drone_velocities)

            self.state[i,2] = self.drone_directions[i]
            
            self.update_drone_positions(i, self.drone_velocities)

            reward = self.reward_grid[int(self.state[i,0]), int(self.state[i,1])]
            print(f"the reward for drone {i} is {reward}")
            reward_N.append(reward)

            if reward == self.goal_reward:
                self.reward_grid[int(self.state[i,0]), int(self.state[i,1])] = 0.5*self.goal_reward


    
    
    '''
    def step(self, action):

        # actions = actions.reshape((self.N, self.k_a))

        self.collective_reward = 0

        # compute turning angles from given actions
        if self.counter == self.max_timesteps:
            # self.collective_reward = 0
            self.counter = 0
            self.truncated = True
            return self.drone_grid_positions, self.collective_reward, self.done, self.truncated, {"max steps reached"}

        
        self.new_angles = np.zeros((self.N))
        for i in range(self.N):
            
            action_angle = self.compute_angles(action)

            self.update_drone_velocities(i, self.action_angle)

            self.update_drone_directions(i, self.drone_velocities)

            self.drone_grid_positions = self.update_drone_positions(i, self.drone_grid_positions, self.drone_velocities)


            reward_index = np.argwhere(self.drone_grid_positions==1)[0]
            reward_drone_i = self.reward_grid[reward_index[0],reward_index[1]]

            self.collective_reward += reward_drone_i

        self.counter += 1

        return self.drone_grid_positions, self.collective_reward, self.done, self.truncated, {"continue"}
        #return observation, reward, terminated, truncated, info
    '''

        
    '''
    def reset(self, seed=None, options=None):
        
        self.collective_reward = 0
        self.done = False
        self.truncated = False
        self.counter = 0

        self.initialize_grid()
        
        self.drone_grid_positions = np.zeros((self.L, self.L))
        self.initialize_drones()


        return self.drone_grid_positions
    '''
        

    #def render(self):


    def render(self):
        
        fig, ax = plt.subplots(figsize = (10,10))
        a=1/(self.L)


        patch_A = plt.Polygon([[a*(self.origin_Ax), a*(self.origin_Ay)], [a*(self.origin_Ax+La_x), a*(self.origin_Ay)], [a*(self.origin_Ax+La_x), a*(self.origin_Ay+La_y)], [a*(self.origin_Ax), a*(self.origin_Ay+La_y)] ], fc = 'lightblue', zorder=5)
        ax.add_patch(patch_A)


        # patch_B = plt.Polygon([[a*(self.origin_Bx), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By+Lb_y)], [a*(self.origin_Bx), a*(self.origin_By+Lb_y)] ], fc = 'lightgreen')
        # ax.add_patch(patch_B)
        for i in np.argwhere(self.reward_grid==self.goal_reward):
            
            patch_B = plt.Rectangle(a*i, width = a, height = a, fc = 'lightgreen', zorder=6)
            ax.add_patch(patch_B)
        
        for i in np.argwhere(self.reward_grid==(0.5*self.goal_reward)):
            
            patch_B = plt.Rectangle(a*i, width = a, height = a, fc = 'lightgreen', zorder=6, alpha=0.5)
            ax.add_patch(patch_B)

        

        boundary_X0 = plt.Polygon([[a*0, a*0], [a*self.boundary_width, a*0], [a*self.boundary_width, a*self.L], [a*0, a*self.L] ], fc = 'black', zorder=4)
        boundary_Xend = plt.Polygon([[a*(self.L-self.boundary_width), a*0], [a*self.L, a*0], [a*self.L, a*self.L], [a*(self.L-self.boundary_width), a*self.L] ], fc = 'black', zorder=4)
        boundary_Y0 = plt.Polygon([[a*0, a*0], [a*self.L, a*0], [a*self.L, a*self.boundary_width], [a*0, a*self.boundary_width] ], fc = 'black', zorder=4)
        boundary_Yend = plt.Polygon([[a*0, a*(self.L-self.boundary_width)], [a*self.L, a*(self.L-self.boundary_width)], [a*self.L, a*self.L], [a*0, a*self.L] ], fc = 'black', zorder=4)
        ax.add_patch(boundary_X0)
        ax.add_patch(boundary_Xend)
        ax.add_patch(boundary_Y0)
        ax.add_patch(boundary_Yend)

        # Draw grid
        for x in range(self.L):
            for y in range(self.L):
                pos=(a*x, a*y)
                width=a
                lattice = plt.Rectangle( pos, width, width, fc='none', ec='black', linewidth=0.2, zorder=13 )
                ax.add_patch(lattice)


        # Draw drones on grid
        for i in range(self.N):
                
                patch_vision = plt.Circle((a*self.state[i,0], a*self.state[i,1]), self.Rv*a, zorder=9, fc = "darkorchid", alpha=0.3)
                ax.add_patch(patch_vision)
                patch_drone = plt.Circle((a*self.state[i,0], a*self.state[i,1]), 0.5*a, fc = 'darkblue', zorder=10)
                ax.add_patch(patch_drone)
                patch_drone_dir = plt.arrow(a*self.state[i,0], a*self.state[i,1], a*self.drone_velocities[i,0], a*self.drone_velocities[i,1], color='red', zorder=11)
                ax.add_patch(patch_drone_dir)



        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')
        plt.show()

        #plt.pause(0.1)





    def close(self):
        self.drone_grid_positions = None

if __name__ == "__main__":

    N = 1
    k_a = 5
    k_s = 16
    theta_max = np.pi / 4
    boundary_width = 1
    Rv = 3
    L = 20 + (2 * boundary_width)
    La_x = 5
    La_y = 10
    Lb_x = 5
    Lb_y = 20
    origin_Ax = 0 + boundary_width
    origin_Ay = 5 + boundary_width
    origin_Bx = L - Lb_x - boundary_width
    origin_By = 0 + boundary_width
    max_timesteps = 100
    step_reward = 0
    goal_reward = 1

    n_timesteps = 20

    settings = {"N": N,
                "k_a": k_a,
                "k_s": k_s,
                "theta_max": theta_max,
                "boundary_width": boundary_width,
                "L": L,
                "Rv": Rv,
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
    env.render()
    #print(np.argwhere(env.reward_grid==goal_reward)[:,0])
    for t in range(n_timesteps):
        actions = [1]
        env.step(actions)
        env.render()
    #env.reset()
    # update_drone_positions(self, i, velocities)
    # env.render()
    #positions = 
    #env.update_drone_positions(self, i, positions, velocities)
    
    '''
    for i in range(n_timesteps):

        actions = np.array([[0,1,0,0,0,0],[0,0,1,0,0,0]])
        env.step(actions)

        


        env.render()
        
    #env.render()
    '''
