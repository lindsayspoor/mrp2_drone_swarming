import numpy as np
from QLearning import QLearning
from Environment_Drones import EnvironmentDrones
import matplotlib.pyplot as plt
import tqdm

def q_learning_cycle(N, n_timesteps, episode_length, env_settings, qlearning_settings, plot=False):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 

    env = EnvironmentDrones(env_settings)
    agent = QLearning(qlearning_settings)
    #costs = [] # should be defined for all N agents
    timestep_rewards = np.zeros((n_timesteps))
    episode_rewards = []
    positions = np.zeros((N, 2, n_timesteps))

    state, observations = env.reset()

    t = 0
    while t < n_timesteps:
        #for i in range(N):

        actions = agent.epsgreedy_action_selection(observations)
        next_state, next_observations, reward, done = env.step(actions)
        #costs.append(cost)
        timestep_rewards[t] = reward
        positions[:,:,t] = env.positions
        agent.update(observations,actions,reward,next_observations)

        if done:
            state, observations = env.reset()
            episode_rewards.append(reward)
        else:
            observations = next_observations

        t += 1


        if plot:
            env.render(Q_table=agent.Q_table,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    plt.plot(env.alignment_costs)
    plt.show()
    print(agent.Q_table[0])
    #return cost # a list with all the cost value observed at each timestep
    return timestep_rewards, episode_rewards, positions
	



def test():
    
    n_timesteps = 10000
    gamma = 0.9#1.0
    learning_rate = 0.01
    k_a = 7#20
    N = 10
    dim = 2
    k_s = 21
    L = 10
    periodic = False
    v0 = 1
    R =  1
    episode_length = 1000

    # Exploration
    #policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.05
    #temp = 1.0



    # Plotting parameters
    plot = False


    # environment settings
    env_settings = {"dim":dim,
                    "N":N,
                    "L":L,
                    "periodic": periodic,
                    "v0": v0,
                    "R": R,
                    "k_a": k_a,
                    "k_s": k_s,
                    "max_steps": episode_length
                    }

    # qlearning settings
    qlearning_settings = {"S": k_s,
                        "A": k_a,
                        "learning_rate": learning_rate,
                        "gamma": gamma,
                        "epsilon": epsilon,
                        "N": N
                        }

    timestep_rewards, episode_rewards, positions = q_learning_cycle(N, n_timesteps, episode_length, env_settings, qlearning_settings, plot)
    #print("Obtained timestep rewards: {}".format(timestep_rewards))
    #print("Obtained episode rewards: {}".format(episode_rewards))


    plt.figure()
    plt.plot(np.arange(n_timesteps), timestep_rewards)
    plt.xlabel("timesteps")
    plt.ylabel("reward")
    plt.title("Timestep rewards")
    plt.show()
    '''
    plt.figure()
    plt.plot(np.arange(len(episode_rewards)), episode_rewards)
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.title("Episode rewards")
    plt.show()

    '''
    plt.figure()
    for i in range(N):
        plt.plot(positions[i,0,:100],positions[i,1,:100])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Drone trajectory")
    plt.show()

if __name__ == '__main__':
    test()