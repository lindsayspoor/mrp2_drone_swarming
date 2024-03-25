import numpy as np
from QLearning import QLearning
from Environment_Drones import EnvironmentDrones


def q_learning_cycle(N, n_timesteps, env_settings, qlearning_settings, plot=False):
    ''' runs a single repetition of q_learning
    Return: rewards, a vector with the observed rewards at each timestep ''' 

    env = EnvironmentDrones(env_settings)
    agent = QLearning(qlearning_settings)
    #costs = [] # should be defined for all N agents
    rewards = np.zeros((env_settings["N"], n_timesteps))


    state = env.reset()
    t = 0
    while t < n_timesteps:
        for i in range(N):

            action = agent.epsgreedy_action_selection(state, i)
            next_state,reward,done = env.step(action)
            #costs.append(cost)
            rewards[i,t] = reward
            agent.update(i, state,action,reward,next_state)

            if done:
                state=env.reset()
            else:
                state=next_state

        t+=1


        if plot:
            env.render(Q_table=agent.Q_table,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Q-learning execution

    #return cost # a list with all the cost value observed at each timestep
    return rewards
	



def test():
    
    n_timesteps = 50000
    gamma = 1.0
    learning_rate = 0.01
    k = 8
    N = 10

    # Exploration
    #policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    #temp = 1.0



    # Plotting parameters
    plot = False


    # environment settings
    env_settings = {"dim":2,
                    "N":N,
                    "L":10,
                    "periodic": False,
                    "v0": 1,
                    "R": 1,
                    "k": k
                    }

    # qlearning settings
    qlearning_settings = {"S": k,
                        "A": k,
                        "learning_rate": learning_rate,
                        "gamma": gamma,
                        "epsilon": epsilon,
                        "N": N
                        }

    rewards = q_learning_cycle(N, n_timesteps, env_settings, qlearning_settings, plot)
    print("Obtained rewards: {}".format(rewards))

if __name__ == '__main__':
    test()