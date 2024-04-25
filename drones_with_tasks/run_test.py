import numpy as np
from Environment_task1 import Env_Task1
#from elegantrl.agents.AgentMADDPG import AgentMADDPG
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


N = 1
k_a = 5
k_s = 16
theta_max = np.pi / 4
boundary_width = 1
Rv = 3
L = 20 + (2 * boundary_width)
La_x = 5
La_y = 10
Lb_x = 5
Lb_y = 20
origin_Ax = 0 + boundary_width
origin_Ay = 5 + boundary_width
origin_Bx = L - Lb_x - boundary_width
origin_By = 0 + boundary_width
max_timesteps = 200
step_reward = 0.03
goal_reward = 10

n_timesteps = 10000000
eval_eps = 10000

settings = {"N": N,
            "k_a": k_a,
            "k_s": k_s,
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
check_env(env)

model = PPO("MlpPolicy", env=env, verbose=1)
model.learn(total_timesteps=n_timesteps, progress_bar=True)
model.save("test_model")
# model.load("test_model")

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=eval_eps, deterministic=True, render=False, return_episode_rewards=False)
obs, info = env.reset()
eps_reward = []
eps_reward_i = 0
for i in range(eval_eps):
    action = model.predict(obs)
    obs, rewards, dones, trunc, info = env.step(action)
    eps_reward_i += rewards
    # env.render()
    if dones:
        eps_reward.append(eps_reward_i)
        eps_reward_i = 0
        obs, info=env.reset()
        print(f"resetting env")

mean_reward = np.mean(eps_reward)
std_reward = np.std(eps_reward)

    


print(f"Mean reward per episode is {mean_reward} +/- {std_reward}")


# agent_class = AgentMADDPG