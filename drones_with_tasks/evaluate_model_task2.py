import gym
import numpy as np
from PPOAgent import Agent
from utils import plot_learning_curve
from DronesEnv_task2 import Env_Task2
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 3
    M = N-1
    k_a = 3
    k_s = 4
    k_l = 5
    # theta_max = np.pi / 4
    theta_max  = np.pi /2
    boundary_width = 1
    Rv = 15
    L = 12 + (2 * boundary_width)
    La_x = 10
    La_y = 10
    Lb_x = 10
    Lb_y = 10
    origin_Ax = 1 + boundary_width
    origin_Ay = 1 + boundary_width
    origin_Bx = L - Lb_x - boundary_width - 1
    origin_By = 1 + boundary_width
    max_timesteps = 100
    step_reward = 0
    goal_reward = 1
    boundary_reward = -1
    reward_decay = 0.5

    step_counter = 20
    batch_size = 5 # actual length of each batch is step_counter/batch_size amount of batches, so for each learning epoch it constructs step_counter/batch_size amount of batches.
    n_epochs = 4 # and then for each epoch it samples multiple batches of length batch_size out of the memory and performs learning on it.
    alpha = 0.0003
    n_training_games = 1000


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
                "goal_reward": goal_reward,
                "boundary_reward": boundary_reward,
                "reward_decay": reward_decay
                }
    
    env = Env_Task2(settings=settings)



    render = False


    filename = f"task2_{n_training_games=}_{N=}_{Rv=}_{step_counter=}_{batch_size=}_{n_epochs=}_{alpha=}_{max_timesteps=}_{step_reward=}_{goal_reward=}_{boundary_reward=}_{k_a=}_{k_l=}_{k_s=}_{theta_max=}"

    agent = Agent(n_agents = N, n_actions=k_a, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=(env.observation_space.shape), filename=filename)


    n_evaluation_games = 100
 
    best_score = np.min(env.reward_grid)# env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    grid_visits = []
    agent.load_models()

    for i in tqdm(range(n_evaluation_games)):
        observation_N = env.reset()
        if render:
            env.render()
        done_N = [False] * N
        score = 0
        while not any(done_N):
            # loop through all drones
            action_N = []

            for j in range(N):
                action, prob, val = agent.choose_action(observation_N[j])
                # print(f"{action=}")
                action_N.append(action)

            observation_N_, reward_N, done_N, info_N = env.step(action_N)

            if render:
                env.render()
            # print(f"{reward_N=}")
            n_steps += 1
            score += reward_N[0]

            observation_N = observation_N_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])



        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
    
        grid_visits_i = env.grid_visits
        grid_visits.append(grid_visits_i)

    mean_grid_visits = np.mean(np.array(grid_visits), axis=0)
    # print(f"{mean_grid_visits.shape}")
    plt.figure()
    plt.imshow(mean_grid_visits, cmap = "coolwarm")
    plt.colorbar()
    plt.xlabel("x grid positions")
    plt.ylabel("y grid positions")
    plt.title("Mean grid visits per game")
    plt.savefig(f"plots/grid_visits_task2_{n_evaluation_games=}_{n_training_games=}_{N=}_{Rv=}_{step_counter=}_{batch_size=}_{n_epochs=}_{alpha=}_{max_timesteps=}_{step_reward=}_{goal_reward=}_{boundary_reward=}_{k_a=}_{k_l=}_{k_s=}_{theta_max=}.pdf")





# evaluate based on path of drones to see how the exploration pattern in area B has evolved.
