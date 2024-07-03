import numpy as np
from stable_baselines3 import PPO
from utils import plot_learning_curve, plot_log_results
from DronesEnv_combi import DronesEnv_Combined
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

    N = 4
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
    swarm_factor = 0
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

    n_eval_episodes = 1
    render = True

    
    

    # Create log dir

    log_dir = f"log_dir_combined_N{N}_{swarm_factor=}/"
    # log_dir = f"log_dir_combined_N{N}/"
    check_freq = 1000
    order_param_check = 10000
    model_path = f"best_model_combined_task_{n_episodes=}_{N=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{swarm_factor=}_{collision_factor=}_{compactness_const=}_{reward_decay=}_2"
    # model_path = "task2_N1_100maxsteps"
    os.makedirs(log_dir, exist_ok=True)

    env = DronesEnv_Combined(settings=settings, render_mode='rgb_array')

   
    # model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range, policy_kwargs={"net_arch":dict(pi=[n_nodes]*n_layers, vf=[n_nodes]*n_layers)})
    model = PPO.load(log_dir+model_path)

    episode_rewards = []
    order_params=[]
    grid_visits=[]

    for i in tqdm(range(n_eval_episodes)):
        obs, info = env.reset()
        if render:
            env.render()
        done = False
        trunc = False
        episode_reward = 0
        
        while not trunc:
            action, _ = model.predict(obs)
            obs, reward, done, trunc, info = env.step(action)

            if render:
                env.render()

            episode_reward += reward

        episode_rewards.append(episode_reward)
        order_params.append(env.order_param)
        grid_visits_i = env.grid_visits
        grid_visits.append(grid_visits_i)



    mean_order_param=np.mean(order_params)
    mean_reward = np.mean(episode_rewards)
    mean_grid_visits = np.mean(np.array(grid_visits), axis=0)


    # print()
    # print(game_grid_visits)

    print(f"{mean_reward=}")
    print(f"{mean_order_param=}")


    if n_eval_episodes == 1:
        plt.figure()
        plt.imshow(mean_grid_visits, cmap = "coolwarm", origin='lower')
        plt.plot([env.origin_Bx-(1/2), env.origin_Bx-(1/2)], [env.origin_By-(1/2), env.origin_By+env.Lb_y-(1/2)], color='limegreen', linewidth=1 )
        plt.plot([env.origin_Bx-(1/2), env.origin_Bx+env.Lb_x-(1/2)], [env.origin_By-(1/2), env.origin_By-(1/2)], color='limegreen', linewidth=1 )
        plt.plot([env.origin_Bx+env.Lb_x-(1/2), env.origin_Bx+env.Lb_x-(1/2)], [env.origin_By-(1/2), env.origin_By+env.Lb_y-(1/2)], color='limegreen', linewidth=1 )
        plt.plot([env.origin_Bx-(1/2), env.origin_Bx+env.Lb_x-(1/2)], [env.origin_By+env.Lb_y-(1/2), env.origin_By+env.Lb_y-(1/2)], color='limegreen', linewidth=1 )
        for (i,j), label in np.ndenumerate(np.reshape(mean_grid_visits, (L, L))):
            plt.text(j,i,f"{label:.2f}",ha='center',va='center', size="xx-small")
        plt.colorbar()
        plt.xlabel("x grid positions")
        plt.ylabel("y grid positions")
        plt.title("Mean grid visits per game")
        plt.savefig(f"plots/combined_task_grid_visits_{n_eval_episodes=}_{n_episodes=}_{N=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{swarm_factor=}_{collision_factor=}_{compactness_const=}.pdf")

    else:
        plt.figure()
        plt.imshow(mean_grid_visits, cmap = "coolwarm", origin='lower')
        for (i,j), label in np.ndenumerate(np.reshape(mean_grid_visits, (L, L))):
            plt.text(j,i,f"{label:.2f}",ha='center',va='center', size="xx-small")
        plt.colorbar()
        plt.xlabel("x grid positions")
        plt.ylabel("y grid positions")
        plt.title("Mean grid visits per game")
        plt.savefig(f"plots/combined_task_grid_visits_{n_eval_episodes=}_{n_episodes=}_{N=}_{Rv=}_{n_steps=}_{batch_size=}_{n_epochs=}_{lr=}_{ent_coef=}_{clip_range=}_{max_timesteps=}_{swarm_factor=}_{collision_factor=}_{compactness_const=}.pdf")







