import numpy as np
import tensorflow as tf
from tf_agents.networks import q_network

class RLNetwork(q_network):
    def __init__(self, in_dim, out_dim):
        self.weights = np.zeros(in_dim)


