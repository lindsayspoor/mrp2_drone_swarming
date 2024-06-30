import numpy as np
from stable_baselines3 import PPO
from utils import plot_learning_curve, plot_log_results
from DronesEnv_task2sb3 import Env_Task2
from tqdm import tqdm
import os
from stable_baselines3.common.monitor import Monitor
from CallbackClass import PlottingCallback, SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt
import matplotlib.animation as animation


if __name__ == "__main__":

    N = 1
    M = N-1
    k_a = 3
    k_s = 4
    k_l = 5
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
    reward_decay = 0.75
    
    n_episodes = 80000
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    lr = 0.00001
    ent_coef = 0.001
    clip_range = 0.2

    n_layers=3
    n_nodes=128

    n_eval_episodes = 1
    render = True

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
    

    # Create log dir
    log_dir = f"log_dir_N1_agent3/"
    # model_path = f"best_model_task2_n_episodes=80000_N=1_3_128_Rv=15_n_steps=2048_batch_size=64_n_epochs=10_lr=1e-05_ent_coef=0.001_clip_range=0.2_max_timesteps=100_step_reward=0_goal_reward=1_boundary_reward=-1_reward_decay=0.75_k_a=3_k_l=5_k_s=4_2"
    model_path = f"best_model_task2_{n_episodes=}_{N=}_{n_layers}_{n_nodes}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{step_reward=}_{goal_reward=}_{boundary_reward=}_{reward_decay=}_{k_a=}_{k_l=}_{k_s=}_2"
    os.makedirs(log_dir, exist_ok=True)

    env = Env_Task2(settings=settings)
   
    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range, policy_kwargs={"net_arch":dict(pi=[n_nodes]*n_layers, vf=[n_nodes]*n_layers)})
    model = PPO.load(log_dir+model_path)

    episode_rewards = []
    grid_visits = []


    for i in tqdm(range(n_eval_episodes)):
        obs, info = env.reset()
        if render:
            env.render()
        done = False
        trunc = False
        n = 0
        episode_reward = 0
        
        while not trunc:
            action, _ = model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)
            if n == (N-1):
                if render:
                    env.render()
                    n = 0
                episode_reward += reward
            else:
                n +=1
        episode_rewards.append(episode_reward)
        grid_visits_i = env.grid_visits
        grid_visits.append(grid_visits_i)


    
    mean_reward = np.mean(episode_rewards)
    mean_grid_visits = np.mean(np.array(grid_visits), axis=0)

    # print((np.array(grid_visits)>0))
    game_grid_visits = np.mean((np.array(grid_visits)>0), axis=0)
    # print()
    # print(game_grid_visits)

    print(f"{mean_reward=}")





    plt.figure()
    plt.imshow(mean_grid_visits, cmap = "coolwarm", origin='lower')
    plt.plot([origin_Ax-(1/2), origin_Ax-(1/2)], [origin_Ay-(1/2), origin_Ay+La_y-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax-(1/2), origin_Ax+La_x-(1/2)], [origin_Ay-(1/2), origin_Ay-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax+La_x-(1/2), origin_Ax+La_x-(1/2)], [origin_Ay-(1/2), origin_Ay+La_y-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax-(1/2), origin_Ax+La_x-(1/2)], [origin_Ay+La_y-(1/2), origin_Ay+La_y-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax-(1/2)-1, origin_Ax-(1/2)-1], [origin_Ay-(1/2)-1, origin_Ay+La_y-(1/2)+1], color='red', linewidth=1 )
    plt.plot([origin_Ax-(1/2)-1, origin_Ax+La_x-(1/2)+1], [origin_Ay-(1/2)-1, origin_Ay-(1/2)-1], color='red', linewidth=1 )
    plt.plot([origin_Ax+La_x-(1/2)+1, origin_Ax+La_x-(1/2)+1], [origin_Ay-(1/2)-1, origin_Ay+La_y-(1/2)+1], color='red', linewidth=1 )
    plt.plot([origin_Ax-(1/2)-1, origin_Ax+La_x-(1/2)+1], [origin_Ay+La_y-(1/2)+1, origin_Ay+La_y-(1/2)+1], color='red', linewidth=1 )
    for (i,j), label in np.ndenumerate(np.reshape(mean_grid_visits, (L, L))):
        plt.text(j,i,f"{label:.2f}",ha='center',va='center', size="xx-small")
    plt.colorbar()
    plt.xlabel("x grid positions")
    plt.ylabel("y grid positions")
    plt.title("Mean grid visits per game")
    plt.savefig(f"plots/grid_visits_task2_{n_episodes=}_{N=}_{n_layers}_{n_nodes}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{step_reward=}_{goal_reward=}_{boundary_reward=}_{reward_decay=}_{k_a=}_{k_l=}_{k_s=}.pdf")




    plt.figure()
    plt.imshow(game_grid_visits, cmap = "coolwarm", origin='lower')
    plt.plot([origin_Ax-(1/2), origin_Ax-(1/2)], [origin_Ay-(1/2), origin_Ay+La_y-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax-(1/2), origin_Ax+La_x-(1/2)], [origin_Ay-(1/2), origin_Ay-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax+La_x-(1/2), origin_Ax+La_x-(1/2)], [origin_Ay-(1/2), origin_Ay+La_y-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax-(1/2), origin_Ax+La_x-(1/2)], [origin_Ay+La_y-(1/2), origin_Ay+La_y-(1/2)], color='limegreen', linewidth=1 )
    plt.plot([origin_Ax-(1/2)-1, origin_Ax-(1/2)-1], [origin_Ay-(1/2)-1, origin_Ay+La_y-(1/2)+1], color='red', linewidth=1 )
    plt.plot([origin_Ax-(1/2)-1, origin_Ax+La_x-(1/2)+1], [origin_Ay-(1/2)-1, origin_Ay-(1/2)-1], color='red', linewidth=1 )
    plt.plot([origin_Ax+La_x-(1/2)+1, origin_Ax+La_x-(1/2)+1], [origin_Ay-(1/2)-1, origin_Ay+La_y-(1/2)+1], color='red', linewidth=1 )
    plt.plot([origin_Ax-(1/2)-1, origin_Ax+La_x-(1/2)+1], [origin_Ay+La_y-(1/2)+1, origin_Ay+La_y-(1/2)+1], color='red', linewidth=1 )
    for (i,j), label in np.ndenumerate(np.reshape(game_grid_visits, (L, L))):
        plt.text(j,i,f"{label:.2f}",ha='center',va='center', size="xx-small")
    plt.colorbar()
    plt.xlabel("x grid positions")
    plt.ylabel("y grid positions")
    plt.title("Mean game visits per grid tile")
    plt.savefig(f"plots/game_grid_visits_task2_{n_episodes=}_{N=}_{n_layers}_{n_nodes}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{step_reward=}_{goal_reward=}_{boundary_reward=}_{reward_decay=}_{k_a=}_{k_l=}_{k_s=}.pdf")







