import numpy as np
from cgitb import small
from typing import Callable
from matplotlib.pyplot import grid
from tqdm import tqdm
import numpy as np

from model import Model, Actions

import matplotlib.pyplot as plt
from world_config import cliff_world, small_world, grid_world
from plot_vp import plot_vp


def value_iteration(model: Model, n_episodes: int = 100, epsilon: float = 1e-6, async_update: bool = False):
    """
    Perform Value Iteration on the given model.

    Parameters:
    - model: Instance of the Model class representing the environment.
    - n_episodes: Maximum number of iterations.
    - epsilon: Convergence threshold for stopping criteria.
    - async_update: If True, perform asynchronous value updates.

    Returns:
    - V: Optimal value function.
    - pi: Optimal policy.
    - iterations: Number of iterations performed.
    """
    V = np.zeros(model.num_states)  # Initialize value function
    pi = np.zeros(model.num_states, dtype=int)  # Initialize policy
    convergence_curve = []
    iterations = 0

    for iteration in range(n_episodes):
        delta = 0
        iterations += 1

        if async_update:
            # Asynchronous Value Iteration: Update V in-place
            for s in model.states:
                if s == model.fictional_end_state:
                    continue

                v_old = V[s]
                q_values = [
                    sum(
                        model.transition_probability(s, s_next, a) * (model.reward(s, a) + model.gamma * V[s_next])
                        for s_next in model.states
                    ) for a in Actions
                ]
                V[s] = max(q_values)
                pi[s] = np.argmax(q_values)
                delta = max(delta, abs(v_old - V[s]))
        else:
            # Synchronous Value Iteration: Use a temporary copy of V
            V_new = np.copy(V)
            for s in model.states:
                if s == model.fictional_end_state:
                    continue

                q_values = [
                    sum(
                        model.transition_probability(s, s_next, a) * (model.reward(s, a) + model.gamma * V[s_next])
                        for s_next in model.states
                    ) for a in Actions
                ]
                V_new[s] = max(q_values)
                pi[s] = np.argmax(q_values)

            delta = np.max(np.abs(V - V_new))
            V = V_new

        convergence_curve.append(delta)

        if delta < epsilon:
            break

    return V, pi, convergence_curve, iterations


# if __name__ == "__main__":

#     model = Model(cliff_world)

#     # Run Synchronous Value Iteration
#     V_vi, pi_vi = value_iteration(model, async_update=False)
#     plot_vp(model, V_vi, pi_vi)
#     plt.show()

#     # Run Asynchronous Value Iteration
#     V_avi, pi_avi = value_iteration(model, async_update=True)
#     plot_vp(model, V_avi, pi_avi)
#     plt.show()

#     # # Compare policies and analyze complexity
#     # policy_vs_value_iteration(model)
