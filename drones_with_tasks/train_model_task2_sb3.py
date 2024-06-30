import jax.numpy as np
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
    
    n_episodes = 40000
    n_steps = 20
    batch_size = 5
    n_epochs = 4
    lr = 0.00001
    ent_coef = 0.001
    clip_range = 0.2

    n_layers=3
    n_nodes=64


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
    model_path = f"task2_{n_episodes=}_{N=}_{n_layers}_{n_nodes}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{step_reward=}_{goal_reward=}_{boundary_reward=}_{reward_decay=}_{k_a=}_{k_l=}_{k_s=}"
    # model_path = "task2_N1_100maxsteps"
    os.makedirs(log_dir, exist_ok=True)

    env = Env_Task2(settings=settings)
    env = Monitor(env, log_dir)

    plotting_callback = PlottingCallback(log_dir=log_dir)
    auto_save_callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_path=model_path)

    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range, policy_kwargs={"net_arch":dict(pi=[n_nodes]*n_layers, vf=[n_nodes]*n_layers)})
    print(model.policy)
    model.learn(total_timesteps = N*max_timesteps*n_episodes, callback=auto_save_callback, progress_bar=True)
    model.save(f"models/pposb3_task2_"+model_path+".zip")

    # plot_results([log_dir], N*max_timesteps*n_episodes, results_plotter.X_EPISODES, "Test")
    # plt.show()
    plot_log_results(log_dir, model_path)


