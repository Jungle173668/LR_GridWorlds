import numpy as np
import matplotlib.pyplot as plt
from world_config import small_world
from sarsa import sarsa
from expected_sarsa import expected_sarsa

from tqdm import tqdm

from plot_vp import plot_vp
from model import Model, Actions


def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(results, param_name):
    plt.figure(figsize=(12, 8))

    for param_value, cum_r in results.items():
        smoothed_cum_r = moving_average(cum_r, window_size=50)
        plt.plot(smoothed_cum_r, label=f"{param_name}={param_value}")

    plt.xlabel('Episodes',fontsize = 17)
    plt.ylabel('Smoothed Cumulative Reward', fontsize = 17)
    plt.title(f'SARSA: Impact of {param_name} on Learning', fontsize = 17)
    plt.legend(fontsize = 14)
    plt.show()

def plot_comparison(cum_r_sarsa, cum_r_expected_sarsa):
    plt.figure(figsize=(10, 6))

    smoothed_cum_r_sarsa = moving_average(cum_r_sarsa, window_size=50)
    smoothed_cum_r_expected_sarsa = moving_average(cum_r_expected_sarsa, window_size=50)

    plt.plot(smoothed_cum_r_sarsa, label='SARSA', color='blue')
    plt.plot(smoothed_cum_r_expected_sarsa, label='Expected SARSA', color='green')

    plt.xlabel('Episodes', fontsize=14)
    plt.ylabel('Smoothed Cumulative Reward', fontsize=14)
    plt.title('Comparison of SARSA vs Expected SARSA', fontsize=14)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    model = Model(small_world)
    '''
    # find optimum params'
    '''
    # a_values = [0.01, 0.05, 0.1, 0.5, 0.9]  # learning rate alpha
    # epsilon_values = [0.01, 0.05, 0.1, 0.3, 0.5, 0.9]  # exploration params

    # maxit_values = [5, 10, 20, 50, 100, 300,]  # maximum number of iterations per episode
    # # maxit_values = [100]
    # # n_episodes_values = [100, 1000, 5000, 10000]  # number of episodes
    # n_episodes_values = [1000]

    # for param_name, param_values in [("alpha", a_values), ("epsilon", epsilon_values), ("maxit", maxit_values), ("n_episodes", n_episodes_values)]:
    #     results = {}
    #     for param_value in param_values:
    #         kwargs = {param_name: param_value}
    #         V, pi, cum_r, cum_iter = sarsa(model, **kwargs)
    #         results[param_value] = cum_r
    #     plot_results(results, param_name)

    # V, pi, cum_r, cum_iter = sarsa(model)

    # plot_vp(model, V, pi)
    # plt.show()

    # V, pi, cum_r, cum_iter = sarsa(model)

    '''
    # discuss learned policy
    '''
    # # best params
    # V, pi, cum_r, cum_iter = sarsa(model, maxit=100, alpha=0.5, epsilon=0.05, n_episodes=200)

    # plot_vp(model, V, pi)
    # plt.show()

    # V, pi, cum_r, cum_iter = sarsa(model)

    '''
    Compare sarsa and expected_sarsa
    '''
    V, pi, cum_r_sarsa, cum_iter = sarsa(model, maxit=100, alpha=0.5, epsilon=0.05, n_episodes=500)
    V, pi, cum_r_expected_sarsa, cum_iter = expected_sarsa(model, maxit=100, alpha=0.5, epsilon=0.05, n_episodes=500)


    plot_comparison(cum_r_sarsa, cum_r_expected_sarsa)

