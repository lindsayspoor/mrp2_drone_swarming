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
import gymnasium
from gymnasium.wrappers import RecordVideo
import time


def save_frames_as_gif(frames, path='plots/', filename='gym_animation.gif'):

    #Mess with this to change frame size
    # plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.figure()

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


if __name__ == "__main__":

    N = 2
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
    reward_decay = 0.5

    n_episodes = 15000
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
                "step_reward": step_reward,
                "goal_reward": goal_reward,
                "boundary_reward": boundary_reward,
                "reward_decay": reward_decay
                }
    

    # Create log dir
    video_every=1
    log_dir = "log_dir/"
    model_path = "best_model_task2"
    os.makedirs(log_dir, exist_ok=True)

    env = Env_Task2(settings=settings, render_mode="rgb_array")
    env = RecordVideo(env, "plots/video", episode_trigger = lambda episode_id: (episode_id%video_every)==0)
   
    model = PPO("MlpPolicy", env, verbose=0, n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs, learning_rate=lr, ent_coef=ent_coef, clip_range=clip_range)
    model = PPO.load(log_dir+model_path+".zip")

    env.start_video_recorder()


    frames=[]
    obs, info = env.reset()
    # if render:
    env.render()
    done = False
    trunc = False
    n = 0
    episode_reward = 0
    
    while not trunc:
        action, _ = model.predict(obs)
        obs, reward, done, trunc, info = env.step(action)
        if n == N:
            env.render()
            n = 0
        else:
            n +=1
    
    env.close()
    
    # save_frames_as_gif(frames)
