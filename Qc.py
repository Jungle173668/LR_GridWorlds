import numpy as np
import matplotlib.pyplot as plt
from world_config import small_world
from sarsa import sarsa
from expected_sarsa import expected_sarsa

import sys
from tqdm import tqdm

from plot_vp import plot_vp
from model import Model, Actions
from q_learning import q_learning


def moving_average(data, window_size=50):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_results(results, param_name):
    plt.figure(figsize=(12, 8))

    for param_value, cum_r in results.items():
        smoothed_cum_r = moving_average(cum_r, window_size=50)
        plt.plot(smoothed_cum_r, label=f"{param_name}={param_value}")

    plt.xlabel('Episodes',fontsize=17)
    plt.ylabel('Smoothed Cumulative Reward',fontsize=17)
    plt.title(f'Q-Learning: Impact of {param_name} on Learning',fontsize=17)
    plt.legend(fontsize=14)
    plt.show()

def plot_comparison(cum_r_q_learning, cum_r_sarsa):
    plt.figure(figsize=(10, 6))

    smoothed_q_learning = moving_average(cum_r_q_learning, window_size=50)
    smoothed_sarsa = moving_average(cum_r_sarsa, window_size=50)

    plt.plot(smoothed_q_learning, label='Q-Learning', color='red')
    plt.plot(smoothed_sarsa, label='SARSA', color='blue')

    plt.xlabel('Episodes',fontsize=17)
    plt.ylabel('Smoothed Cumulative Reward',fontsize=17)
    plt.title('Comparison of Q-Learning vs SARSA',fontsize=17)
    plt.legend(fontsize=14)
    plt.show()



if __name__ == "__main__":
    model = Model(small_world)

    '''
    figure out the params
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
    #         V, pi, cum_r, cum_iter = q_learning(model, **kwargs)
    #         results[param_value] = cum_r
    #     plot_results(results, param_name)

    # V_q, pi_q, cum_r_q, cum_iter_q = q_learning(model)
    # # V_sarsa, pi_sarsa, cum_r_sarsa, cum_iter_sarsa = sarsa(model)

    # plot_vp(model, V_q, pi_q)
    # plt.show()
    '''
    compare the algos
    '''
    V_q, pi_q, cum_r_q, cum_iter_q = q_learning(model, maxit=100, alpha=0.5, epsilon=0.05, n_episodes=200)
    V_sarsa, pi_sarsa, cum_r_sarsa, cum_iter_sarsa = sarsa(model, maxit=50, alpha=0.5, epsilon=0.05, n_episodes=200)

    # compare efficiency
    plot_comparison(cum_r_q, cum_r_sarsa)

    # policy
    plot_vp(model, V_q, pi_q)
    plt.show()

    # cum_iter:
    print(f"Total Iterations - Q-Learning: {np.average(cum_iter_q)}, SARSA: {np.average(cum_iter_sarsa)}")
