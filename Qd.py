from Qc import *

import numpy as np
import matplotlib.pyplot as plt
from world_config import small_world, cliff_world
from sarsa import sarsa
from expected_sarsa import expected_sarsa

import sys
from tqdm import tqdm

from plot_vp import plot_vp
from model import Model, Actions
from q_learning import q_learning


model = Model(cliff_world)

V_q, pi_q, cum_r_q, cum_iter_q = q_learning(model, maxit=100, alpha=0.5, epsilon=0.7, n_episodes=500)
# policy
plot_vp(model, V_q, pi_q)
plt.show()

V_sarsa, pi_sarsa, cum_r_sarsa, cum_iter_sarsa = sarsa(model, maxit=50, alpha=0.5, epsilon=0.7, n_episodes=500)
# policy
plot_vp(model, V_sarsa, pi_sarsa)
plt.show()

# compare efficiency
plot_comparison(cum_r_q, cum_r_sarsa)
