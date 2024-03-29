import numpy as np
from Environment_task1 import Env_Task1
from stable_baselines3 import DQN






N = 2
k_a = 6
k_s = 10
theta_max = np.pi / 6
L = 100
La_x = 20
La_y = 20
Lb_x = 20
Lb_y = 20
origin_Ax = 0
origin_Ay = 40
origin_Bx = L-Lb_x
origin_By = 40
max_timesteps = 100
step_reward = -1
goal_reward = 300

n_timesteps = 1000

settings = {"N": N,
            "k_a": k_a,
            "k_s": k_s,
            "theta_max": theta_max,
            "L": L,
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

model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps = n_timesteps, progress_bar = True)

# evaluation after training
'''
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

'''

n_evaluation_steps = 100
obs, info = env.reset()

rewards = []

for i in range(n_evaluation_steps):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    rewards.append(reward)
    if terminated or truncated:
        env.render()
        obs, info = env.reset()    

