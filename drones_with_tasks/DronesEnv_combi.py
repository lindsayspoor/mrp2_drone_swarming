import jax.numpy as np
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Any




class DronesEnv_Combined(gym.Env):
    """"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, settings, render_mode=None):
        super().__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        

        self.N = settings["N"] # N is the numer of drones the swarm consists of
        self.k_a = settings["k_a"] # k_a equally spaced angles for the actions in range [-theta_max, theta_max]
        self.k_s = settings["k_s"] # k_s equally spaced angles for the direction angle in the range [-pi, pi)
        self.k_l = settings["k_l"] # k_l equally spaced observation regions in the range [-pi, pi)
        self.M = settings["M"] # M is the maximum occupation number per observation region
        self.theta_max = settings["theta_max"]
        self.boundary_width = settings["boundary_width"] # number of grid elements the boundary width consists of
        self.L = settings["L"] # size of grid of total enviroment (LxL)
        self.window_size = self.L
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
        self.boundary_reward = settings["boundary_reward"]
        self.goal_reward = settings["goal_reward"]
        self.periodic_boundaries = settings["periodic"]
        self.swarm_factor = settings["swarm_factor"]
        self.collision_factor = settings["collision_factor"]
        self.compactness_const = settings["compactness_const"]
        self.reward_decay = settings["reward_decay"]

        self.action_angles = np.linspace(-self.theta_max, self.theta_max, self.k_a)
        # print(f"{self.action_angles=}")
        # self.direction_angles = np.linspace(0,2*np.pi, self.k_s) 
        
        self.direction_angles = np.linspace(0, 2*np.pi, self.k_s+1)[0:-1]

        self.obs_regions = np.linspace(0, 2*np.pi, self.k_l)
        # print(f"{self.obs_regions=}")

        self.counter = 0 # updates each step to count what timestep we are in
        self.done = False
        self.truncated = False
        self.collective_reward = 0
        # self.rate_loss_neighbors = 0
        self.order_param = 0

        self.action_space = spaces.MultiDiscrete([self.k_a]*self.N) # for all N drones, k_a possible action angles to choose from



        low  = np.zeros((5*self.N))
        high =  np.zeros((5*self.N))
        high[0:(5*self.N):5] = self.L
        high[1:(5*self.N):5] = self.L
        high[2:(5*self.N):5] = self.direction_angles[-1]
        high[3:(5*self.N):5] = self.direction_angles[-1]
        high[4:(5*self.N):5] = self.goal_reward


        self.observation_space = spaces.Box(low = low, high=high, dtype = np.float64)


        # self.initialize_grid()

        # self.grid_visits = np.zeros((self.L, self.L))


        # self.state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle
        # self.old_state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle

        # self.initialize_drones()

        # self.initialize_rewards()


        



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
        self.order_param = 0
        self.state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle
        self.old_state = np.zeros((self.N, 3)) # N drones, x coordinate, y coordinate, k_s angle

        self.initialize_grid()
        self.initialize_drones()
        self.initialize_rewards()
        self.grid_visits = np.zeros((self.L, self.L))


        obs= self.get_obs()


        return obs, {}



    def initialize_rewards(self):
        '''Initialize the rewards per grid cell.'''

        self.reward_grid = np.zeros((self.L,self.L))


        # self.reward_grid[self.grid_B_positions[0][0]:(self.grid_B_positions[-1][0]+1), self.grid_B_positions[0][1]:(self.grid_B_positions[-1][1]+1)] = self.goal_reward
        for x, y in self.grid_B_positions:
            self.reward_grid[x][y] = self.goal_reward


    def classify_drones_in_bins(self, i, connected_drones_i):
        '''Classifies all drones within range Rv of the i-th drone in k_l positional
        bins.'''

        crowdedness_i = np.zeros((self.k_l-1))
        for j in connected_drones_i:


            theta_ij=np.arctan2((self.state[j,1]-self.state[i,1]), (self.state[j,0]-self.state[i,0])) % (2*np.pi)


            obs_regions_new = [(n + self.state[i,2] - np.pi/2) % (2*np.pi) for n in self.obs_regions.copy()]


            for k in range(self.k_l - 1):
                if theta_ij >= (obs_regions_new[k] % (2*np.pi)) and theta_ij < (obs_regions_new[k+1] % (2*np.pi)):

                    crowdedness_i[k] += 1

                    break

        return crowdedness_i



    def compute_ang_diff_angle(self, phi, theta):
        '''Computes the angular difference angle between angles phi and theta in the range (0,2pi] and selects the discretized
        angle from the list of direction angles afterwards.'''


        alpha = phi - theta


        angular_difference_angle = (alpha + 2*np.pi) % (2*np.pi)


        return angular_difference_angle



    def compute_angular_difference(self, i, connected_drones_i):
        '''Computes the average velocity vector Pi and the angular difference angle, i.e. the difference in direction angle of
        the i-th drone and the average direction angle according to Pi from the other drones in the range of the i-th drone.'''


        Pi = np.zeros(2)
        for j in connected_drones_i:
            if j != i:
                Pi += self.drone_velocities[j,:]
        Pi = Pi / Pi.shape[0]

        self.avg_velocities[i,:] = Pi


        phi_i = self.compute_direction(Pi)

        phi_d_i = self.find_nearest(self.direction_angles, phi_i)


        angular_difference_angle_i = self.compute_ang_diff_angle(phi_d_i , self.drone_directions[i])


        return angular_difference_angle_i






    def get_obs(self):



        obs = np.zeros((5*self.N))

        ang_diff_angles = np.zeros(self.N)
        pos_rewards = np.zeros(self.N)
        new_positions = self.state[:,0:2].copy()

        # find drones within the range Rv of drone i

        for i in range(self.N):
            # connected_drones_i = self.find_connected_drones(i, self.state)
            ang_diff_angles[i] = self.compute_angular_difference(i, np.arange(self.N))
            pos_rewards[i] = self.positional_rewards(i, new_positions)

        grid_visits = self.grid_visits.flatten()

        obs[0:(5*self.N):5] = self.state[:,0]
        obs[1:(5*self.N):5] = self.state[:,1]
        obs[2:(5*self.N):5] = self.state[:,2]
        obs[3:(5*self.N):5] = ang_diff_angles
        obs[4:(5*self.N):5] = pos_rewards



        # crowdedness_i = self.classify_drones_in_bins(i, connected_drones_i)

        


        
#
        # obs_i[0:3] = self.state[i,0:3]
        # obs_i[4:(4+self.k_l-1)] = crowdedness_i
        #add observastion of angular difference angle
        # obs_i[3] = angular_difference_angle_i



        return obs


    def initialize_grid(self):

        self.origin_Bx = np.random.randint(0,self.L-self.Lb_x+1)
        self.origin_By = np.random.randint(0,self.L-self.Lb_y+1)

        self.grid_positions = [[i,j] for i in range(self.L) for j in range(self.L)]
        self.grid_A_positions = [[i,j] for i in range(self.origin_Ax, self.origin_Ax+self.La_x) for j in range(self.origin_Ay, self.origin_Ay+self.La_y)] # define area A on grid
        self.grid_B_positions = [[i,j] for i in range(self.origin_Bx, self.origin_Bx+self.Lb_x) for j in range(self.origin_By, self.origin_By+self.Lb_y)] # define area B on grid



    def initialize_drones(self):
        '''Initialize drone positions within area A.'''

        drone_grid_indices = np.random.choice(np.arange(self.La_x*self.La_y), size=self.N, replace=False) # randomly choose initial grid locations for all N drones in area A
        # by initialising the drones on the grid positions and setting replace = False, all drones will never be initialised onto the same grid cell
        # drone_grid_indices = [24, 75]


        self.drone_directions = np.random.choice(self.direction_angles, size = self.N) # choose random initial directions for all drones
        # self.drone_directions = [self.direction_angles[3]]*self.N
        # self.drone_directions = [self.direction_angles[0], self.direction_angles[2]]
        # print(f"{self.drone_directions=}")
        self.state[:,2] = self.drone_directions


        self.drone_velocities = np.zeros((self.N,2))
        self.avg_velocities = np.zeros((self.N,2))

        for i in range(self.N):
            self.state[i, 0:2] =  [self.grid_A_positions[drone_grid_indices[i]][0],self.grid_A_positions[drone_grid_indices[i]][1]]
            
            self.drone_velocities[i,:] = self.compute_velocities(self.drone_directions[i]) # compute the initial velocity vector for all drones based on the given direction angle


        self.update_order_parameter()








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


    def compute_direction(self, vector):
        '''Computes the angle of a vector in the range (0,2pi].'''


        theta = np.arctan2(vector[1],vector[0])% (2*np.pi)

        return theta




    def update_drone_directions(self, i):
        '''Update the directions of the drones according to their velocities.'''

        theta = self.compute_direction(self.drone_velocities[i,:])


        # theta = np.arctan2(self.drone_velocities[i,1],self.drone_velocities[i,0])% (2*np.pi)

        new_angle = self.find_nearest(self.direction_angles, theta)

        
        self.drone_directions[i] = new_angle





  
    def cast_to_grid(self, i, positions):
        '''Casts i-th drone position to grid.'''
        positions[i,0] = positions[i,0] - (positions[i,0] % 1)
        positions[i,1] = positions[i,1] - (positions[i,1] % 1)

        return positions
    

    def update_drone_positions(self, i):
        '''Update the position of the ith drone.'''

        new_pos = self.state[i,0:2] + self.drone_velocities[i,:]

        self.old_state[i,0:2] = self.state[i,0:2]


        self.state[i,0:2] = new_pos

        # if self.periodic_boundaries:

        self.update_periodic_boundaries(i)

        # else:

            # boundary_reward_i = self.bounce_from_boundaries(i)

            # return boundary_reward_i
        
    def update_periodic_boundaries(self,i):


        self.state[i,0:2] = (self.state[i,0:2] - self.L) % self.L



    def bounce_from_boundaries(self,i):
        '''Makes sure that drones that would have been updated to a position outside of the boundaries are brought back inside
        of the map, where the positions are mirrored onto the boundary and the velocity vector + directions are mirrored as well.'''

        boundary_reward_i = 0

        if (self.state[i,0] >= (self.L - self.boundary_width)):
            # self.done=True
            new_pos_x = (self.L - self.boundary_width) - (self.state[i,0] - (self.L - self.boundary_width))
            self.state[i,0] = new_pos_x
            self.drone_velocities[i,0] = self.drone_velocities[i,0] * -1
            boundary_reward_i += self.boundary_reward

            
        if (self.state[i,0] <= self.boundary_width):
            # self.done=True
            new_pos_x = self.boundary_width - (self.state[i,0] - self.boundary_width)
            self.state[i,0] = new_pos_x
            self.drone_velocities[i,0] = self.drone_velocities[i,0] * -1
            boundary_reward_i += self.boundary_reward

        if (self.state[i,1] >= (self.L - self.boundary_width)):
            # self.done=True
            new_pos_y = (self.L - self.boundary_width) - (self.state[i,1] - (self.L - self.boundary_width))
            self.state[i,1] = new_pos_y
            self.drone_velocities[i,1] = self.drone_velocities[i,1] * -1
            boundary_reward_i += self.boundary_reward

        if (self.state[i,1] <= self.boundary_width):
            # self.done=True
            new_pos_y = self.boundary_width - (self.state[i,1] - self.boundary_width)
            self.state[i,1] = new_pos_y
            self.drone_velocities[i,1] = self.drone_velocities[i,1] * -1
            boundary_reward_i += self.boundary_reward
        
        self.drone_velocities[i,:] = self.drone_velocities[i,:] / np.linalg.norm(self.drone_velocities[i,:])

        return boundary_reward_i





    def compute_angles(self, i, actions):
        '''Compute turning angles from given actions.'''

        new_angles = np.zeros((self.N))

        action_index = np.argwhere(actions[i] == 1)[0][0]
        new_angles[i] = self.action_angles[action_index]

        return new_angles
        




    def update_drone_velocities(self, i, angle):
        '''Update the velocities of the i-th drone given the rotation angle.'''




        new_drone_velocities = np.zeros((2))

        new_drone_velocities[0] = self.drone_velocities[i,0]*np.cos(angle) - self.drone_velocities[i,1]*np.sin(angle) 
        new_drone_velocities[1] = self.drone_velocities[i,0]*np.sin(angle) + self.drone_velocities[i,1]*np.cos(angle) 

        self.drone_velocities[i,:] = new_drone_velocities
        # normalize the velocities
        self.drone_velocities[i,:] = self.drone_velocities[i,:] / np.linalg.norm(self.drone_velocities[i,:])



    def update_order_parameter(self):
        '''Computes the order parameter of the entire system.'''

        # self.order_param = np.abs(np.sum(np.sqrt(self.drone_velocities[:,0]**2+self.drone_velocities[:,1]**2)))/self.N

        self.order_param = np.sqrt(np.sum(self.drone_velocities[:,0])**2+np.sum(self.drone_velocities[:,1])**2)/self.N



    def find_connected_drones(self, i, positions):
        '''For the i-th drone find the other drones that are within the connectivity range Rv.'''

        drones_in_range=[]

        for j in range(self.N):
            if j != i:
                if (positions[j,0] < positions[i,0] + self.Rv) and (positions[j,0] > positions[i,0] - self.Rv) and (positions[j,1] < positions[i,1] + self.Rv) and (positions[j,1] > positions[i,1] - self.Rv):
                    drones_in_range.append(j)

        return drones_in_range
    


    def collision_rewards(self, i, positions):
        '''Assign the collision rewards to the i-th drone.'''

        # connected_drones_i = self.find_connected_drones(i, positions)


        collision_reward_i = 0
        for j in np.arange(self.N):
            if j!=i:
                dist_x_ij = (positions[j,0]-positions[i,0] + self.L/2) % self.L - self.L/2
                dist_y_ij = (positions[j,1]-positions[i,1] + self.L/2) % self.L - self.L/2
                if np.sqrt((dist_x_ij)**2+(dist_y_ij)**2) < 1:
                    collision_reward_i -= (self.compactness_const - np.sqrt((dist_x_ij) ** 2+(dist_y_ij) ** 2))


        return collision_reward_i
    
    
    
    def order_param_reward(self):

        self.update_order_parameter()

        return self.order_param
    
    def positional_rewards(self,i,new_positions):
        '''Checks whether the new position update results in a drone flying out of the boundary. Game is terminated in this case.'''


        new_copied_positions = np.array(new_positions).copy()


        reward_positions = self.cast_to_grid(i, new_copied_positions)

        reward_i = self.reward_grid[int(reward_positions[i,0]), int(reward_positions[i,1])]
        if reward_i != 0:
            self.reward_grid[int(reward_positions[i,0]), int(reward_positions[i,1])] *= self.reward_decay

        self.grid_visits[int(reward_positions[i,1]), int(reward_positions[i,0])] +=1


        return reward_i


    def get_rewards(self):
        '''Assigns individual rewards to each drone.'''

        reward_N = []

        new_positions = self.state[:,0:2].copy()
        # old_positions = self.old_state[:,0:2].copy()

        # rate_loss_neighbors = []

        for i in range(self.N):


            # swarming_reward_i = self.swarming_rewards(i, new_positions, old_positions)
            order_param_reward_i = -(1-self.order_param_reward())
            position_reward_i = self.positional_rewards(i, new_positions)
            collision_reward_i = self.collision_rewards(i, new_positions)


            
            # reward_i = swarming_reward_i + collision_reward_i + boundary_rewards[i]
            # if self.periodic_boundaries:

            reward_i = (self.swarm_factor * order_param_reward_i) + (self.collision_factor * collision_reward_i) + position_reward_i
            # else:
                # reward_i = order_param_reward_i + collision_reward_i + boundary_rewards[i]  + position_reward_i


            # print(f"reward_{i} = {order_param_reward_i=} + {collision_reward_i=} + {boundary_rewards[i]=} + {position_reward_i=}")

            reward_N.append(reward_i)

            # rate_loss_neighbors.append(swarming_reward_i)

        # self.rate_loss_neighbors = np.sum(np.array(rate_loss_neighbors)/self.reward_swarm)/self.N

        return reward_N

    
    def step(self, actions):
        '''Step function for all N actions for all N agents.'''

        # self.actions_all[self.i] = action
        
        # first let all drones do their actions and update their positions, directions accordingly

        # if self.i == (self.N-1):

        # boundary_rewards = []

        for i in range(self.N):

            action_angle = self.action_angles[int(actions[i])]


            self.update_drone_velocities(i, action_angle)

            self.update_drone_positions(i)

            # boundary_reward_i = self.update_drone_positions(i)

            # boundary_rewards.append(boundary_reward_i)

            self.update_drone_directions(i)

            self.old_state[i,2] = self.state[i,2]

            self.state[i,2] = self.drone_directions[i]
        
        # print(f"{self.state=}")

        obs = self.get_obs()
        

        reward_N = self.get_rewards()

        self.update_order_parameter()


        collective_reward  = np.sum(reward_N)/self.N

        self.counter+=1
        if self.counter == self.max_timesteps:
            self.truncated = True

        return obs, collective_reward, self.done, self.truncated, {}
        

    





    def render(self):


        fig, ax = plt.subplots(figsize = (10,10))
        a=1/(self.L)


        patch_A = plt.Polygon([[a*(self.origin_Ax), a*(self.origin_Ay)], [a*(self.origin_Ax+self.La_x), a*(self.origin_Ay)], [a*(self.origin_Ax+self.La_x), a*(self.origin_Ay+self.La_y)], [a*(self.origin_Ax), a*(self.origin_Ay+self.La_y)] ], fc = 'lightblue', zorder=3)
        ax.add_patch(patch_A)

        for i in range(self.boundary_width, self.L-self.boundary_width):

            for j in range(self.boundary_width,self.L-self.boundary_width):
                patch_B = plt.Rectangle((a*i, a*j), width = a, height = a, fc = 'darkgreen', zorder=5, alpha= (self.reward_grid[i,j]))
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
                
                # patch_vision = plt.Circle((a*self.state[i,0]+a/2, a*self.state[i,1]+a/2), self.Rv*a, zorder=9, fc = "darkorchid", alpha=0.1)
                # ax.add_patch(patch_vision)
                patch_drone = plt.Circle((a*self.state[i,0]+a/2, a*self.state[i,1]+a/2), 0.5*a, fc = 'darkblue', zorder=10)
                ax.add_patch(patch_drone)
                patch_drone_dir = plt.arrow(a*self.state[i,0]+a/2, a*self.state[i,1]+a/2, a*self.drone_velocities[i,0], a*self.drone_velocities[i,1], color='red', zorder=11)
                ax.add_patch(patch_drone_dir)



        ax.set_aspect(1)
        ax.set_xticks([])
        ax.set_yticks([])




        plt.axis('off')
        plt.savefig(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/plots/render_plots/render_{self.N=}_{self.counter=}.pdf")
        # plt.show()
        plt.close()







    def close(self):
        self.state = None



if __name__ == "__main__":

    N = 4
    M = N-1
    k_a = 3
    k_s = 4
    k_l = 5
    theta_max  = np.pi /2
    boundary_width = 0
    Rv = 10000 # set radius of all drones to infinity (large number)
    L = 20 + (2 * boundary_width)
    La_x = 10
    La_y = 10
    Lb_x = 10
    Lb_y = 10
    origin_Ax = 5 + boundary_width
    origin_Ay = 5 + boundary_width
    origin_Bx = 2 # L - Lb_x - boundary_width - 1
    origin_By = 2 # 1 + boundary_width
    max_timesteps = 2
    boundary_reward = -10
    goal_reward = 1
    periodic = True
    swarm_factor = 1
    collision_factor = 1
    compactness_const = 1
    reward_decay = 0.75


    n_episodes = 1000
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
                "boundary_reward": boundary_reward,
                "goal_reward": goal_reward,
                "periodic": periodic,
                "swarm_factor": swarm_factor,
                "collision_factor": collision_factor,
                "compactness_const": compactness_const,
                "reward_decay": reward_decay
                }
    


    env = DronesEnv_Combined(settings=settings, render_mode='rgb_array')


    obs_0, info = env.reset()
    # print(f"{obs_0=}")
    
    # print(f"{env.order_param=}")
    # print(f"{env.drone_velocities=}")

    env.render()


    for i in range(300):

        actions = np.random.randint(0,k_a, size=N)
        # print(f"{actions=}")
        

        obs, reward, done, trunc, info = env.step(actions)
        # print(f"{reward=}")
        # print(f"{obs=}")
        # print(f"{env.drone_velocities=}")
        # print(f"{env.order_param=}")

        env.render()

        if trunc:
            # episode_id+=1
            obs_0, info = env.reset()
    env.close()



    
