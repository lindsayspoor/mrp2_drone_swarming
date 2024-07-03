import jax.numpy as np
from stable_baselines3 import PPO
from utils import plot_learning_curve, plot_log_results, plot_order_param_training
from DronesEnv_combi import DronesEnv_Combined
from tqdm import tqdm
import os
from stable_baselines3.common.monitor import Monitor
from CallbackClass import PlottingCallback, SaveOnBestTrainingRewardCallback, SaveOnBestTrainingRewardCallbackTask3



if __name__ == "__main__":


    N = 2
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
    max_timesteps = 100
    boundary_reward = -10
    goal_reward = 1
    periodic = True
    swarm_factor = 1
    collision_factor = 1
    compactness_const = 1
    reward_decay = 0.75



    n_episodes = 10000
    n_steps = 2048
    batch_size = 64
    n_epochs = 10
    lr = 0.0001
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
                "boundary_reward": boundary_reward,
                "goal_reward": goal_reward,
                "periodic": periodic,
                "swarm_factor": swarm_factor,
                "collision_factor": collision_factor,
                "compactness_const": compactness_const,
                "reward_decay": reward_decay
                }
    


    
    

    # Create log dir
    log_dir = f"log_dir_combined_N{N}_{swarm_factor=}/"
    check_freq = 1000
    order_param_check = 10000
    model_path = f"combined_task_{n_episodes=}_{N=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{swarm_factor=}_{collision_factor=}_{compactness_const=}_{reward_decay=}"
    # model_path = "task2_N1_100maxsteps"
    os.makedirs(log_dir, exist_ok=True)

    env = DronesEnv_Combined(settings=settings, render_mode='rgb_array')
    env = Monitor(env, log_dir)

    plotting_callback = PlottingCallback(log_dir=log_dir)
    # auto_save_callback = SaveOnBestTrainingRewardCallbackTask3(check_freq=check_freq, order_param_check=order_param_check,log_dir=log_dir, env = env, N=N, model_path=model_path)
    auto_save_callback =  SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_path=model_path)

    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range, policy_kwargs={"net_arch":dict(pi=[n_nodes]*n_layers, vf=[n_nodes]*n_layers)})
    model.learn(total_timesteps = max_timesteps*n_episodes, callback=auto_save_callback, progress_bar=True)
    model.save(f"models/pposb3_"+model_path)

    # plot_results([log_dir], N*max_timesteps*n_episodes, results_plotter.X_EPISODES, "Test")
    # plt.show()
    plot_log_results(log_dir, model_path)

    # plot_order_param_training(log_dir, model_path, auto_save_callback.save_order_param_path, check_freq)


