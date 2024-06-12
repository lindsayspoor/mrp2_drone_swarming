import jax.numpy as np
from stable_baselines3 import PPO
from utils import plot_learning_curve, plot_log_results
from DronesEnv_task3sb3 import Env_Task3
from tqdm import tqdm
import os
from stable_baselines3.common.monitor import Monitor
from CallbackClass import PlottingCallback, SaveOnBestTrainingRewardCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common import results_plotter
import matplotlib.pyplot as plt


if __name__ == "__main__":

    N = 4
    M = N-1
    k_a = 3
    k_s = 4
    k_l = 5
    theta_max  = np.pi /2
    boundary_width = 1
    Rv = 4
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
    boundary_reward = -1
    reward_swarm = -1

    
    n_episodes = 5000
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    lr = 0.0003
    ent_coef = 0.0
    clip_range = 0.2


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
                "reward_swarm": reward_swarm
                }
    

    # Create log dir
    log_dir = f"log_dir_N4/"
    model_path = f"task3_{n_episodes=}_{N=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{boundary_reward=}_{reward_swarm=}_{k_a=}_{k_l=}_{k_s=}_{theta_max=}"
    # model_path = "task2_N1_100maxsteps"
    os.makedirs(log_dir, exist_ok=True)

    env = Env_Task3(settings=settings)
    env = Monitor(env, log_dir)

    plotting_callback = PlottingCallback(log_dir=log_dir)
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_path=model_path)

    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range)
    model.learn(total_timesteps = N*max_timesteps*n_episodes, callback=auto_save_callback, progress_bar=True)
    model.save(f"models/pposb3_task3_"+model_path)

    # plot_results([log_dir], N*max_timesteps*n_episodes, results_plotter.X_EPISODES, "Test")
    # plt.show()
    plot_log_results(log_dir, model_path)


