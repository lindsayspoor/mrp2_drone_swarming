import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.results_plotter import load_results, ts2xy

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)





def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window

    return np.convolve(values, weights, "valid")

def plot_log_results(log_folder,  save_model_path, title="Average training reward"):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), "episodes")

    y = moving_average(y, window=50)

    # Truncate x
    x = x[len(x) - len(y) :]

    np.savetxt(f"{log_folder}{save_model_path}.csv",(x,y) )

    fig = plt.figure(title)
    plt.plot(x, y, color = 'blue', linewidth=0.9)
    plt.yscale("linear")
    plt.xlabel("Number of training episodes")
    plt.ylabel("Reward")
    plt.grid()
    plt.savefig(f'{log_folder}{save_model_path}.pdf')


def plot_order_param_training(log_folder, save_model_path, file_path, check_freq):

    '''Plots the order parameter during training'''

    order_params = np.loadtxt(file_path)

    plt.figure(figsize=(10,8))
    plt.style.use('ggplot')
    plt.plot(np.arange(len(order_params))*check_freq, order_params)
    plt.xlabel("Timestep", fontsize=25)
    plt.ylabel(f"$\Phi$", fontsize=25)
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=19)
    plt.savefig(f"{log_folder}order_param_{save_model_path}.pdf")






