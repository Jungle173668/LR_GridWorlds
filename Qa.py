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

from policy_iteration import policy_iteration
from value_iteration import value_iteration

def compare_policies(policy1, policy2):
    """Compare two policies and return whether they are identical."""
    return np.array_equal(policy1, policy2)

def policy_vs_value_iteration(model):
    # Run Policy Iteration
    V_pi, pi_pi, cone_pi, iter_pi = policy_iteration(model)
    plot_vp(model, V_pi, pi_pi)
    plt.show()

    # Run Synchronous Value Iteration
    V_vi, pi_vi, con_sync, iter_sync = value_iteration(model, async_update=False)
    plot_vp(model, V_vi, pi_vi)
    plt.show()

    # Run Asynchronous Value Iteration
    V_avi, pi_avi, con_async, iter_async = value_iteration(model, async_update=True)
    plot_vp(model, V_avi, pi_avi)
    plt.show()
    
    # print(pi_pi, pi_vi, pi_avi)
    print("Policy Iteration and Synchronous Value Iteration produce same policy:", compare_policies(pi_pi, pi_vi))
    print("Policy Iteration and Asynchronous Value Iteration produce same policy:", compare_policies(pi_pi, pi_avi))

    return V_pi, pi_pi, V_vi, pi_vi, V_avi, pi_avi


def plot_convergence():
    model = Model(grid_world)

    # sync
    V_sync, pi_sync, convergence_sync, iterations_syc = value_iteration(model, async_update=False)
    print(convergence_sync)

    # async
    V_async, pi_async, convergence_async, iterations_async = value_iteration(model, async_update=True)
    print(convergence_async)

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(convergence_sync, label="Synchronous Value Iteration")
    plt.plot(convergence_async, label="Asynchronous Value Iteration")

    plt.xlabel("Iteration")
    plt.ylabel("Max Value Difference (delta)")
    plt.title("Convergence Curve (Delta) Comparison")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison():
    model = Model(cliff_world)  # select world

    # sync
    V_vi, pi_vi, convergence_vi, iterations_vi = value_iteration(model, async_update=False)

    # async
    V_avi, pi_avi, convergence_avi, iterations_avi = value_iteration(model, async_update=True)

    # pi
    V_pi, pi_pi, convergence_pi, iterations_pi = policy_iteration(model)

    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(convergence_vi, label=f"Synchronous VI (Iterations: {iterations_vi})")
    plt.plot(convergence_avi, label=f"Asynchronous VI (Iterations: {iterations_avi})")
    plt.plot(convergence_pi, label=f"Policy Iteration (Iterations: {iterations_pi})")
    plt.xlabel("Iteration")
    plt.ylabel("Max Value Difference (delta)")
    plt.title("Convergence Curve Comparison: PI vs. Sync VI vs. Async VI")
    plt.legend()
    plt.grid(True)
    plt.show()

    # print iteration nums
    print(f"Total Iterations for Synchronous VI: {iterations_vi}")
    print(f"Total Iterations for Asynchronous VI: {iterations_avi}")
    print(f"Total Iterations for PI: {iterations_pi}")



# compare 2 results PI & VI & VIA
model = Model(grid_world)
policy_vs_value_iteration(model)

# plot the convergence curve for sync & async
# plot_convergence()

# compare pi, vi, via
# plot_comparison()

