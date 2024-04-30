import numpy as np
from MADDPG import MADDPG
from ReplayBuffer import ReplayBuffer
from DronesEnvironment import Env_Task1
import torch as T
T.autograd.set_detect_anomaly(True)
import pdb

def obs_list_to_state_vector(observation):
    state = np.array([])
    # print(f"{observation=}")
    for obs in observation:
        # print(f"{np.array(obs).flatten().shape=}")
        state = np.concatenate([state, obs])
    return state



if __name__ == '__main__':
    #scenario = 'simple'
    task = 'task1'

    N = 2
    if N == 1:
        M=1
    else:
        M=N-1
    # M = N-1
    k_a = 5
    k_s = 16
    k_l = 8
    theta_max = np.pi / 4
    boundary_width = 1
    Rv = 3
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
    # n_agents = env.n
    dims_actor = []
    for i in range(N):
        dims_actor.append(k_s * (k_l-1))

    # print(f"{dims_actor=}")
    dims_critic = np.sum(dims_actor, axis=0)

    # action space is a list of arrays, assume each agent has same action space
    N_actions = k_a
    maddpg_agents = MADDPG(dims_actor, dims_critic, N, N_actions, 
                           layer1_nodes=64, layer2_nodes=64,  
                           alpha1=0.01, alpha2=0.01, task=task,
                           directory='checkpoints/')

    memory = ReplayBuffer(N, N_actions, 1000000, 1024, dims_actor, dims_critic)


    PRINT_INTERVAL = 500
    N_GAMES = 50000
    MAX_STEPS = 25
    total_steps = 0
    score_history = []
    evaluate = False
    best_score = 0

    if evaluate:
        maddpg_agents.load()

    for i in range(N_GAMES):
        obs, info = env.reset()
        score = 0
        done = [False] * N
        episode_step = 0
        while not any(done):
            if evaluate:
                env.render()
                #time.sleep(0.1) # to slow down the action for the video
            # print(f"{obs[0]=}")
            # exit()
            actions = maddpg_agents.select_action(obs)
            # print(f"{actions=}")
            obs_, reward, done, info = env.step(actions)

            state = obs_list_to_state_vector(obs)
            state_ = obs_list_to_state_vector(obs_)

            if episode_step >= MAX_STEPS:
                done = [True]* N


            memory.store_transition(obs, state, actions, reward, obs_, state_, done)

            if total_steps % 100 == 0 and not evaluate:
                # print(f"{total_steps=}")
                maddpg_agents.learn(memory)
            # print("learn executed")

            obs = obs_

            score += sum(reward)
            total_steps += 1
            episode_step += 1

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        if not evaluate:
            if avg_score > best_score:
                maddpg_agents.save()
                best_score = avg_score
        if i % PRINT_INTERVAL == 0 and i > 0:
            print('episode', i, 'average score {:.1f}'.format(avg_score))