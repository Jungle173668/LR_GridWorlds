from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions

import matplotlib.pyplot as plt
from world_config import cliff_world, small_world, grid_world
from plot_vp import plot_vp


def policy_iteration(model: Model, maxit: int = 100):

    V = np.zeros((model.num_states,))  # values
    pi = np.zeros((model.num_states,))  # actions
    convergence_curve = []
    iterations = 0

    # value in state s
    def compute_value(s, a, reward: Callable):
        return np.sum(
            [
                model.transition_probability(s, s_, a)
                * (reward(s, a) + model.gamma * V[s_])
                for s_ in model.states
            ]
        )

    def policy_evaluation():
        delta = 0
        for s in model.states:
            R = model.reward(s, pi[s])  
            v_old = V[s]
            V[s] = compute_value(s, pi[s], lambda *_: R)
            delta = max(delta, abs(v_old - V[s]))  
        return delta

    def policy_improvement():
        for s in model.states:
            action_index = np.argmax(
                [compute_value(s, a, model.reward) for a in Actions]
            )
            pi[s] = Actions(action_index)

    
    convergence_curve.append(policy_evaluation())
    for i in tqdm(range(maxit)):
        # internal iterations
        delta = 0

        for _ in range(3):
            delta = policy_evaluation()

        # delta = policy_evaluation()

        # record the curve
        convergence_curve.append(delta)

        # improve the policy
        pi_old = np.copy(pi)
        policy_improvement()

        # break when no change
        if all(pi_old == pi):
            print("Policy Iteration converged.")
            break
        
        iterations += 1

    return V, pi, convergence_curve, iterations


# if __name__ == "__main__":

#     model = Model(cliff_world)
#     V, pi = policy_iteration(model)
#     plot_vp(model, V, pi)
#     plt.show()

    # model = Model(small_world)
    # V, pi = policy_iteration(model)
    # plot_vp(model, V, pi)
    # plt.show()

    # model = Model(grid_world)
    # V, pi = policy_iteration(model)
    # plot_vp(model, V, pi)
    # plt.show()
