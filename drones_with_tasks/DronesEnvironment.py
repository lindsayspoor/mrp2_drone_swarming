import numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any
# from marllib import marl



class Env_Task1(gym.Env):
    """Environment for drones to perform task 1, which is go from area A to area B as a collective swarm of N drones."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, settings):
        super().__init__()

        self.N = settings["N"] # N is the numer of drones the swarm consists of
        self.k_a = settings["k_a"] # k_a equally spaced angles for the actions in range [-theta_max, theta_max]
        self.k_s = settings["k_s"] # k_s equally spaced angles for the direction angle in the range [-pi, pi)
        self.k_l = settings["k_l"] # k_l equally spaced observation regions in the range [-pi, pi)
        self.M = settings["M"] # M is the maximum occupation number per observation region
        self.theta_max = settings["theta_max"]
        self.boundary_width = settings["boundary_width"] # number of grid elements the boundary width consists of
        self.L = settings["L"] # size of grid of total enviroment (LxL)
        self.Rv = settings["Rv"] # visibility Radius for each drone
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
        # self.obs_regions = np.linspace(-np.pi+(2*np.pi/self.k_s), np.pi, self.k_l)
        self.obs_regions = np.linspace(0, 2*np.pi, self.k_l)
        # print(f"{self.obs_regions=}")
        # exit()
        self.counter = 0 # updates each step to count what timestep we are in
        self.done = False
        self.truncated = False
        self.collective_reward = 0

        self.action_space = spaces.MultiDiscrete([self.k_a] * self.N) # for all N drones, k_a possible action angles to choose from
        # self.observation_space = spaces.MultiDiscrete([[[self.M] * self.k_s] * (self.k_l-1)] * self.N)
        # self.observation_space = spaces.MultiDiscrete([[self.M] * (self.k_l-1)] * self.N) # for each drone we can observe a maximum of M=N-1 other drones per k_l segment (crowdedness)
        # print(np.array([self.observation_space[0].shape, self.observation_space[1].shape]))
        low = np.zeros((self.k_l-1+3))
        low[0:3] = [0,0,self.direction_angles[0]]
        high = np.zeros((self.k_l-1+3))
        high[0:3] = [self.L, self.L, self.direction_angles[-1]]
        high[3:] = self.N
        self.observation_space = spaces.Box(low = low, high=high, dtype = np.float64)
        # print(f"{self.observation_space.shape}")
        # print(f"{self.observation_space.sample()=}")
        # print(np.sum(np.array([self.observation_space[0].shape, self.observation_space[1].shape])))
        # exit()
        self.initialize_grid()

        self.state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle

        self.initialize_drones()

        self.initialize_rewards()
        
        # self.reset()


    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**32
        return [seed1, seed2]


    def reset(self, **kwargs):
        '''Reset function to reset drone states, observations, reward grid back to the initial conditions.'''
        super().reset(**kwargs)

        self.done = False
        self.truncated = False
        self.counter = 0
        self.collective_reward = 0

        # self.state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle

        self.initialize_drones()

        self.initialize_rewards()


        obs_N = self.get_obs()
        # info_N = {}
        # print(f"{obs_N=}")

        return obs_N #, info_N





    def classify_drones_in_bins(self, i, connected_drones_i):
        '''Classifies all drones within range Rv of the i-th drone in k_l positional
        bins.'''

        crowdedness_i = np.zeros((self.k_l-1))
        for j in connected_drones_i:
            if (self.state[j,0]-self.state[i,0]) > 0:
                theta_ij = np.arctan((self.state[j,1]-self.state[i,1])/(self.state[j,0]-self.state[i,0]))
            if (self.state[j,0]-self.state[i,0]) < 0:
                theta_ij = np.arctan((self.state[j,1]-self.state[i,1])/(self.state[j,0]-self.state[i,0])) + np.pi
            if (self.state[j,0]-self.state[i,0]) == 0 and (self.state[j,1]-self.state[i,1]) > 0:
                theta_ij = np.pi/2
            if (self.state[j,0]-self.state[i,0]) == 0 and (self.state[j,1]-self.state[i,1]) < 0:
                theta_ij = (3*np.pi)/2
            
            theta_ij = theta_ij % (2*np.pi)

            # print(f"for drone {i} the angle wrt drone {j} is {theta_ij=}")
            # print(f"shifting with {self.state[i,2] - np.pi/2}")
            # print(f"{(self.obs_regions.copy() + self.state[i,2] - np.pi/2)=}")
            obs_regions_new = [(n + self.state[i,2] - np.pi/2) % (2*np.pi) for n in self.obs_regions.copy()]
            # print(f"{obs_regions_new=}")

            for k in range(self.k_l - 1):
                if theta_ij >= (obs_regions_new[k] % (2*np.pi)) and theta_ij < (obs_regions_new[k+1] % (2*np.pi)):
                    # print(f"{k=} for drone {i} wrt drone {j}")
                    crowdedness_i[k] += 1
                    # print(f"{crowdedness_i=}")
                    break
            # print(f"continuing loop")
        return crowdedness_i




    def get_obs(self):


        observations_N = []

        for i in range(self.N):
            obs_i = np.zeros((self.k_l-1+3))

            # find drones within the range Rv of drone i
            connected_drones_i = self.find_connected_drones(i)
            # print(f"for drone {i}, {connected_drones_i=}")

            crowdedness_i = self.classify_drones_in_bins(i, connected_drones_i)

            obs_i[0:3] = self.state[i,0:3]
            obs_i[3:] = crowdedness_i

            observations_N.append(obs_i)
        # print(f"{observations_N=}")
        return observations_N


    def initialize_grid(self):

        self.grid_positions = [[i,j] for i in range(self.L) for j in range(self.L)]
        self.grid_A_positions = [[i,j] for i in range(self.origin_Ax, self.origin_Ax+self.La_x) for j in range(self.origin_Ay, self.origin_Ay+self.La_y)] # define area A on grid
        self.grid_B_positions = [[i,j] for i in range(self.origin_Bx, self.origin_Bx+self.Lb_x) for j in range(self.origin_By, self.origin_By+self.Lb_y)] # define area B on grid



    def initialize_drones(self):
        '''Initialize drone positions within area A.'''

        drone_grid_indices = np.random.choice(np.arange(self.La_x*self.La_y), size=self.N, replace=False) # randomly choose initial grid locations for all N drones in area A
        # by initialising the drones on the grid positions and setting replace = False, all drones will never be initialised onto the same grid cell

        self.drone_directions = np.random.choice(self.direction_angles, size = self.N) # choose random initial directions for all drones
        self.state[:,2] = self.drone_directions


        self.drone_velocities = np.zeros((self.N,2))

        for i in range(self.N):
            self.state[i, 0:2] =  [self.grid_A_positions[drone_grid_indices[i]][0],self.grid_A_positions[drone_grid_indices[i]][1]]
            
            self.drone_velocities[i,:] = self.compute_velocities(self.drone_directions[i]) # compute the initial velocity vecotr for all drones based on the given direction angle



    def reward_function_linear(self):
        '''Constructs linearly increasing reward function to grid.'''

        for i in range(self.boundary_width, self.L - self.boundary_width):
            self.reward_grid[i,:] = i * self.step_reward


        return self.reward_grid



    def initialize_rewards(self):
        '''Initialize the rewards per grid cell.'''

        self.reward_grid = np.zeros((self.L,self.L))

        self.reward_grid = self.reward_function_linear()

        self.reward_grid[self.grid_B_positions[0][0]:(self.grid_B_positions[-1][0]+1), self.grid_B_positions[0][1]:(self.grid_B_positions[-1][1]+1)] = self.goal_reward

        # define boundaries by assigning a very large negative reward to these grid positions
        self.reward_grid[0, :] = -100
        self.reward_grid[:,0] = -100
        self.reward_grid[-1,:] = -100
        self.reward_grid[:,-1] = -100






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


    def update_drone_directions(self, i):
        '''Update the directions of the drones according to their velocities.'''

        if self.drone_velocities[i,0] == 0:
            new_angle = self.find_nearest(self.direction_angles, 0)
        else:
            new_angle = self.find_nearest(self.direction_angles, np.arctan(self.drone_velocities[i,1]/self.drone_velocities[i,0]))

        
        self.drone_directions[i] = new_angle



    def positional_rewards(self,i,positions):
        '''Checks whether the new position update results in a drone flying out of the boundary. Game is terminated in this case.'''
        
        if positions[i,0] <= self.boundary_width or positions[i,0] >= (self.L-self.boundary_width) or positions[i,1] <= self.boundary_width or positions[i,1] >= (self.L-self.boundary_width):
            reward = -100
            # done = True
            return reward#, done
        


        else:

            copied_positions = np.array(positions).copy()
            reward_positions = self.cast_to_grid(i, copied_positions)

            reward = self.reward_grid[int(reward_positions[i,0]), int(reward_positions[i,1])]
            # done = False
            return reward#, done





    def cast_to_grid(self, i, positions):
        '''Casts i-th drone position to grid.'''
        positions[i,0] = positions[i,0] - (positions[i,0] % 1)
        positions[i,1] = positions[i,1] - (positions[i,1] % 1)

        return positions
    

    def update_drone_positions(self, i):
        '''Update the position of the ith drone.'''

        new_pos = self.state[i,0:2] + self.drone_velocities[i,:]


        self.state[i,0:2] = new_pos




    def compute_angles(self, i, actions):
        '''Compute turning angles from given actions.'''

        new_angles = np.zeros((self.N))

        action_index = np.argwhere(actions[i] == 1)[0][0]
        new_angles[i] = self.action_angles[action_index]

        return new_angles
        




    def update_drone_velocities(self, i, angle):
        '''Update the velocities of the i-th drone given the rotation angle.'''




        self.drone_velocities[i,0] = self.drone_velocities[i,0]*np.cos(angle) - self.drone_velocities[i,1]*np.sin(angle) 
        self.drone_velocities[i,1] = self.drone_velocities[i,0]*np.sin(angle) + self.drone_velocities[i,1]*np.cos(angle) 

        # normalize the velocities
        self.drone_velocities[i,:] = self.drone_velocities[i,:] / np.linalg.norm(self.drone_velocities[i,:])




    def find_connected_drones(self, i):
        '''For the i-th drone find the other drones that are within the connectivity range Rv.'''

        drones_in_range=[]

        for j in range(self.N):
            if j != i:
                if (self.state[j,0] < self.state[i,0] + self.Rv) and (self.state[j,0] > self.state[i,0] - self.Rv) and (self.state[j,1] < self.state[i,1] + self.Rv) and (self.state[j,1] > self.state[i,1] - self.Rv):
                    drones_in_range.append(j)

        return drones_in_range
    


    def collision_rewards(self, i, positions):
        '''Assign the collision rewards to the i-th drone.'''

        connected_drones_i = self.find_connected_drones(i)


        collision_reward_i = 0
        for j in connected_drones_i:

            if np.sqrt((positions[j,0]-positions[i,0])**2+(positions[j,1]-positions[i,1])**2) < 1:
                collision_reward_i -= (1 - np.sqrt((positions[j,0]-positions[i,0])**2+(positions[j,1]-positions[i,1])**2))


        return collision_reward_i
    

    def get_rewards(self):
        '''Assigns individual rewards to each drone.'''

        reward_N = []
        # reward = self.reward_grid[int(self.state[i,0]), int(self.state[i,1])]
        new_positions = self.state[:,0:2].copy()


        for i in range(self.N):

            position_reward_i = self.positional_rewards(i, new_positions)


            collision_reward_i = self.collision_rewards(i, new_positions)

            # TODO: add another term for the swarming here

            reward_i = position_reward_i + collision_reward_i

            
            reward_N.append(reward_i)

            if position_reward_i == self.goal_reward:
                self.reward_grid[int(self.state[i,0]), int(self.state[i,1])] = 0.5*self.goal_reward

        return reward_N

    
    def step(self, actions):
        '''Step function for all N actions for all N agents.'''

        obs_N = []
        reward_N = []
        # done_N = []
        info_N = {'N':[]}


        # first let all drones do their actions and update their positions, directions accordingly
        for i in range(self.N):

            action_angle = self.action_angles[actions[i]]

            self.update_drone_velocities(i, action_angle)

            self.update_drone_directions(i)

            self.state[i,2] = self.drone_directions[i]
            
            self.update_drone_positions(i)
            



        obs_N = self.get_obs()
        # print(f"{obs_N=}")

        reward_N = self.get_rewards()
        # print(f"{reward_N=}")


        collective_reward  = np.sum(reward_N)
        reward_N = [collective_reward] * self.N

        self.counter+=1
        if self.counter == self.max_timesteps:
            self.done = True

        info_N = {}
        trunc_N = False

        done_N = [self.done] * self.N

        return obs_N, reward_N, done_N, info_N
        






    def render(self):

        fig, ax = plt.subplots(figsize = (10,10))
        a=1/(self.L)


        patch_A = plt.Polygon([[a*(self.origin_Ax), a*(self.origin_Ay)], [a*(self.origin_Ax+self.La_x), a*(self.origin_Ay)], [a*(self.origin_Ax+self.La_x), a*(self.origin_Ay+self.La_y)], [a*(self.origin_Ax), a*(self.origin_Ay+self.La_y)] ], fc = 'lightblue', zorder=5)
        ax.add_patch(patch_A)


        for i in range(self.boundary_width,self.origin_Bx):

            for j in range(self.boundary_width,self.L-self.boundary_width):
                patch_B = plt.Rectangle((a*i, a*j), width = a, height = a, fc = 'darkgreen', zorder=3, alpha=(self.reward_grid[i,j]))
                ax.add_patch(patch_B)

        # patch_B = plt.Polygon([[a*(self.origin_Bx), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By)], [a*(self.origin_Bx+Lb_x), a*(self.origin_By+Lb_y)], [a*(self.origin_Bx), a*(self.origin_By+Lb_y)] ], fc = 'lightgreen')
        # ax.add_patch(patch_B)
        for i in np.argwhere(self.reward_grid==self.goal_reward):
            
            patch_B = plt.Rectangle(a*i, width = a, height = a, fc = 'darkgreen', zorder=6)
            ax.add_patch(patch_B)
        
        for i in np.argwhere(self.reward_grid==(0.5*self.goal_reward)):
            
            patch_B = plt.Rectangle(a*i, width = a, height = a, fc = 'darkgreen', zorder=6, alpha=0.5)
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
        self.state = None



if __name__ == "__main__":

    N = 5
    M = N-1
    k_a = 5
    k_s = 16
    k_l = 8
    theta_max = np.pi / 4
    boundary_width = 1
    Rv = 6
    L = 20 + (2 * boundary_width)
    La_x = 5
    La_y = 10
    Lb_x = 5
    Lb_y = 20
    origin_Ax = 1 + boundary_width
    origin_Ay = 5 + boundary_width
    origin_Bx = L - Lb_x - boundary_width
    origin_By = 0 + boundary_width
    max_timesteps = 100
    step_reward = 0.03
    goal_reward = 10

    n_timesteps = 100000
    eval_eps = 100

    settings = {"N": N,
                "k_a": k_a,
                "k_s": k_s,
                "k_l": k_l,
                "M": M,
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

    obs_N, info_N = env.reset()
    env.render()
    for i in range(30):

        actions = np.random.randint(0,k_a, size=N)

        obs_N, reward, done, info_N = env.step(actions)
        # print(f"{reward=}")
        env.render()
        if any(done):
            print(f"{done=}")
            obs_N, info_N = env.reset()


    
