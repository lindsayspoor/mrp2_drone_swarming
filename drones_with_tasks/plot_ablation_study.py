import numpy as np
import matplotlib.pyplot as plt




def plot_learning_rate(data0003, data0001, data001,data00001,data005, x, saving_path, filename):

    average_learning_curve0003 = np.average(data0003, axis=0)
    std_learning_curve0003 = np.std(data0003, axis=0)

    average_learning_curve0003 = moving_average(average_learning_curve0003, window=500)
    std_learning_curve0003 = moving_average(std_learning_curve0003, window=500)

    average_learning_curve0001 = np.average(data0001, axis=0)
    std_learning_curve0001 = np.std(data0001, axis=0)

    average_learning_curve0001 = moving_average(average_learning_curve0001, window=500)
    std_learning_curve0001 = moving_average(std_learning_curve0001, window=500)

    average_learning_curve001 = np.average(data001, axis=0)
    std_learning_curve001 = np.std(data001, axis=0)

    average_learning_curve001 = moving_average(average_learning_curve001, window=500)
    std_learning_curve001 = moving_average(std_learning_curve001, window=500)

    average_learning_curve00001 = np.average(data00001, axis=0)
    std_learning_curve00001 = np.std(data00001, axis=0)

    average_learning_curve00001 = moving_average(average_learning_curve00001, window=500)
    std_learning_curve00001 = moving_average(std_learning_curve00001, window=500)

    average_learning_curve005 = np.average(data005, axis=0)
    std_learning_curve005 = np.std(data005, axis=0)

    average_learning_curve005 = moving_average(average_learning_curve005, window=500)
    std_learning_curve005 = moving_average(std_learning_curve005, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve0003) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    # plt.grid(color='white')
    plt.plot(average_learning_curve00001, label=r"$\alpha=0.00001$")
    plt.fill_between(x, average_learning_curve00001-std_learning_curve00001, average_learning_curve00001+std_learning_curve00001, alpha=0.5)
    plt.plot(average_learning_curve0003, label=r"$\alpha=0.0003$")
    plt.fill_between(x, average_learning_curve0003-std_learning_curve0003, average_learning_curve0003+std_learning_curve0003, alpha=0.5)
    plt.plot(average_learning_curve0001, label=r"$\alpha=0.0001$")
    plt.fill_between(x, average_learning_curve0001-std_learning_curve0001, average_learning_curve0001+std_learning_curve0001, alpha=0.5)
    plt.plot(average_learning_curve001, label=r"$\alpha=0.001$")
    plt.fill_between(x, average_learning_curve001-std_learning_curve001, average_learning_curve001+std_learning_curve001, alpha=0.5)
    plt.plot(average_learning_curve005, label=r"$\alpha=0.005$")
    plt.fill_between(x, average_learning_curve005-std_learning_curve005, average_learning_curve005+std_learning_curve005, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19 )
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")
    # plt.show()


def plot_ent_coef(data0 ,data001, data005, data01, x, saving_path, filename): #, data001, data005,data01,data05, x, saving_path, filename):

    average_learning_curve0 = np.average(data0, axis=0)
    std_learning_curve0 = np.std(data0, axis=0)

    average_learning_curve0 = moving_average(average_learning_curve0, window=500)
    std_learning_curve0 = moving_average(std_learning_curve0, window=500)

    average_learning_curve001 = np.average(data001, axis=0)
    std_learning_curve001 = np.std(data001, axis=0)

    average_learning_curve001 = moving_average(average_learning_curve001, window=500)
    std_learning_curve001 = moving_average(std_learning_curve001, window=500)

    average_learning_curve005 = np.average(data005, axis=0)
    std_learning_curve005 = np.std(data005, axis=0)

    average_learning_curve005 = moving_average(average_learning_curve005, window=500)
    std_learning_curve005 = moving_average(std_learning_curve005, window=500)

    average_learning_curve01 = np.average(data01, axis=0)
    std_learning_curve01 = np.std(data01, axis=0)

    average_learning_curve01 = moving_average(average_learning_curve01, window=500)
    std_learning_curve01 = moving_average(std_learning_curve01, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve0) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(average_learning_curve0, label=r"$c_2=0.0$")
    plt.fill_between(x, average_learning_curve0-std_learning_curve0, average_learning_curve0+std_learning_curve0, alpha=0.5)
    plt.plot(average_learning_curve005, label=r"$c_2=0.005$")
    plt.fill_between(x, average_learning_curve005-std_learning_curve005, average_learning_curve005+std_learning_curve005, alpha=0.5)
    plt.plot(average_learning_curve01, label=r"$c_2=0.01$")
    plt.fill_between(x, average_learning_curve01-std_learning_curve01, average_learning_curve01+std_learning_curve01, alpha=0.5)
    plt.plot(average_learning_curve001, label=r"$c_2=0.001$")
    plt.fill_between(x, average_learning_curve001-std_learning_curve001, average_learning_curve001+std_learning_curve001, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19 )
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")
    # plt.show()



def plot_clip_range(data1, data2, data3, x, saving_path, filename): #, data001, data005,data01,data05, x, saving_path, filename):

    average_learning_curve1 = np.average(data1, axis=0)
    std_learning_curve1 = np.std(data1, axis=0)

    average_learning_curve1 = moving_average(average_learning_curve1, window=500)
    std_learning_curve1 = moving_average(std_learning_curve1, window=500)

    average_learning_curve2 = np.average(data2, axis=0)
    std_learning_curve2 = np.std(data2, axis=0)

    average_learning_curve2 = moving_average(average_learning_curve2, window=500)
    std_learning_curve2 = moving_average(std_learning_curve2, window=500)

    average_learning_curve3 = np.average(data3, axis=0)
    std_learning_curve3 = np.std(data3, axis=0)

    average_learning_curve3 = moving_average(average_learning_curve3, window=500)
    std_learning_curve3 = moving_average(std_learning_curve3, window=500)

    # Truncate x
    x = x[len(x) - len(average_learning_curve2) :]

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(average_learning_curve1, label=r"$\epsilon=0.1$")
    plt.fill_between(x, average_learning_curve1-std_learning_curve1, average_learning_curve1+std_learning_curve1, alpha=0.5)
    plt.plot(average_learning_curve2, label=r"$\epsilon=0.2$")
    plt.fill_between(x, average_learning_curve2-std_learning_curve2, average_learning_curve2+std_learning_curve2, alpha=0.5)
    plt.plot(average_learning_curve3, label=r"$\epsilon=0.3$")
    plt.fill_between(x, average_learning_curve3-std_learning_curve3, average_learning_curve3+std_learning_curve3, alpha=0.5)
    plt.legend(fontsize=21)
    plt.xlabel("Episode", fontsize=25)
    plt.ylabel("Reward", fontsize=25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.savefig(f"{saving_path}{filename}.pdf")
    # plt.show()

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, "valid")




directory1 = "log_dir_N1_agent1/"
directory2 = "log_dir_N1_agent2/"
directory3 = "log_dir_N1_agent3/"



saving_path = "/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/hyperparameters/"

filename_clip1="task2_n_episodes=80000_N=1_Rv=15_n_steps=2048_batch_size=64_n_epochs=10_lr=1e-05_ent_coef=0.001_clip_range=0.1_max_timesteps=100_step_reward=0_goal_reward=1_boundary_reward=-1_reward_decay=0.75_k_a=3_k_l=5_k_s=4_theta_max=1.5707963267948966.csv"
filename_clip2="task2_n_episodes=80000_N=1_Rv=15_n_steps=2048_batch_size=64_n_epochs=10_lr=1e-05_ent_coef=0.001_clip_range=0.2_max_timesteps=100_step_reward=0_goal_reward=1_boundary_reward=-1_reward_decay=0.75_k_a=3_k_l=5_k_s=4_theta_max=1.5707963267948966.csv"
filename_clip3="task2_n_episodes=80000_N=1_Rv=15_n_steps=2048_batch_size=64_n_epochs=10_lr=1e-05_ent_coef=0.001_clip_range=0.3_max_timesteps=100_step_reward=0_goal_reward=1_boundary_reward=-1_reward_decay=0.75_k_a=3_k_l=5_k_s=4_theta_max=1.5707963267948966.csv"

agent_1_x, agent_1_clip1 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory1}{filename_clip1}")
agent_2_x, agent_2_clip1 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory2}{filename_clip1}")
agent_3_x, agent_3_clip1 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory3}{filename_clip1}")

agent_1_x, agent_1_clip2 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory1}{filename_clip2}")
agent_2_x, agent_2_clip2 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory2}{filename_clip2}")
agent_3_x, agent_3_clip2 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory3}{filename_clip2}")

agent_1_x, agent_1_clip3 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory1}{filename_clip3}")
agent_2_x, agent_2_clip3 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory2}{filename_clip3}")
agent_3_x, agent_3_clip3 = np.loadtxt(f"/Users/lindsayspoor/Library/Mobile Documents/com~apple~CloudDocs/Documents/Studiedocumenten/2023-2024/MSc Research Project 2/Code/mrp2_drone_swarming/drones_with_tasks/{directory3}{filename_clip3}")

data_clip1 = np.array([agent_1_clip1, agent_2_clip1, agent_3_clip1])
data_clip2 = np.array([agent_1_clip2, agent_2_clip2, agent_3_clip2])
data_clip3 = np.array([agent_1_clip3, agent_2_clip3, agent_3_clip3])


filename_save_clip = "clip_range_exploration_task_N1"


plot_clip_range(data_clip1, data_clip2, data_clip3, agent_1_x, saving_path, filename_save_clip)
