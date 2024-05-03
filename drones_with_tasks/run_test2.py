import gym
import numpy as np
from PPOAgent import Agent
from utils import plot_learning_curve
from DronesEnvironment import Env_Task1
from tqdm import tqdm

if __name__ == '__main__':
    # env = gym.make('CartPole-v0')
    N = 2
    
    M = 1
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

    # n_timesteps = 100000
    # eval_eps = 100

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


    train = True
    evaluate = True
    render = True
    load = True

    step_counter = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0003
    agent = Agent(n_actions=k_a, batch_size=batch_size,
                  alpha=alpha, n_epochs=n_epochs,
                  input_dims=(env.observation_space.shape))
    n_training_games = 10
    n_evaluation_games = 3

    learning_curve_file = 'plots/test2.pdf'
 
    best_score = np.min(env.reward_grid)# env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    if train:  

        for i in tqdm(range(n_training_games)):
            observation_N = env.reset()
            done_N = [False] * N
            score = 0
            while not any(done_N):
                # loop through all drones
                action_N = []
                prob_N = []
                val_N = []
                # reward_N = []
                # done_N = []
                for j in range(N):
                    action, prob, val = agent.choose_action(observation_N[j])
                    # print(f"{action=}")
                    action_N.append(action)
                    prob_N.append(prob)
                    val_N.append(val)
                observation_N_, reward_N, done_N, info_N = env.step(action_N)
                # print(f"{reward_N=}")
                n_steps += 1
                score += reward_N[0]
                for j in range(N):
                    agent.store_transition(observation_N[j], action_N[j],
                                        prob_N[j], val_N[j], reward_N[j], done_N[j])
                if n_steps % step_counter == 0:
                    agent.learn()
                    learn_iters += 1
                observation_N = observation_N_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, learning_curve_file)

    if evaluate:
        if load:
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





