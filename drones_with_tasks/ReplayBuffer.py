import numpy as np


class ReplayBuffer:
    def __init__(self, N, N_actions, buffer_size, batch_size, dims_actor, dims_critic):
        self.N = N # number of agents
        self.buffer_size = buffer_size # size of the replay buffer
        self.N_actions = N_actions # number of actions
        self.batch_size = batch_size # size of each batch
        self.dims_actor = dims_actor # number of dimensions of actor network
        self.dims_critic = dims_critic # number of dimensions of critic network
        self.counter = 0

        # initialize buffer for critic & actor networks
        self.initialize_critic_buffer()
        self.initialize_actor_buffer()


    def initialize_critic_buffer(self):

        self.critic_state_mem = np.zeros((self.buffer_size, self.dims_critic))
        self.new_critic_state_mem = np.zeros((self.buffer_size, self.dims_critic))
        self.critic_reward_mem = np.zeros((self.buffer_size, self.N))
        self.critic_done_mem  = np.zeros((self.buffer_size, self.N), dtype=bool)


    def initialize_actor_buffer(self):

        self.actor_obs_mem = []
        self.new_actor_obs_mem = []
        self.actor_action_mem = []

        for i in range(self.N):
            self.actor_obs_mem.append(np.zeros((self.buffer_size, self.dims_actor[i])))
            self.new_actor_obs_mem.append(np.zeros((self.buffer_size, self.dims_actor[i])))
            self.actor_action_mem.append(np.zeros((self.buffer_size, self.N_actions)))

    def store_transition(self, observation, state, action, reward, new_observation, new_state, done):

        k = self.counter % self.buffer_size

        for i in range(self.N):
            # print(f"{observation[i]=}")
            self.actor_obs_mem[i][k] = observation[i]
            self.new_actor_obs_mem[i][k] = new_observation[i]
            self.actor_action_mem[i][k] = action[i]

        self.critic_state_mem[k] = state
        self.new_critic_state_mem[k] = new_state
        self.critic_reward_mem[k] = reward
        self.critic_done_mem[k] = done
        self.counter += 1

    
    def sample(self):

        if self.counter < self.buffer_size:

            samples = np.random.choice(self.counter, self.batch_size, replace=False)

        else:

            samples = np.random.choice(self.buffer_size, self.batch_size, replace=False)

        critic_states_batch = self.critic_state_mem[samples]
        critic_rewards_batch = self.critic_reward_mem[samples]
        critic_new_states_batch = self.new_critic_state_mem[samples]
        critic_dones_batch = self.critic_done_mem[samples]


        actor_obs_batch = []
        actor_new_obs_batch = []
        actions_batch = []

        for i in range(self.N):
            actor_obs_batch.append(self.actor_obs_mem[i][samples])
            actor_new_obs_batch.append(self.new_actor_obs_mem[i][samples])
            actions_batch.append(self.actor_action_mem[i][samples])

        return actor_obs_batch, critic_states_batch, actions_batch, critic_rewards_batch, actor_new_obs_batch, critic_new_states_batch, critic_dones_batch
    

    def is_ready_to_sample(self):
        if self.counter >= self.batch_size:
            return True












