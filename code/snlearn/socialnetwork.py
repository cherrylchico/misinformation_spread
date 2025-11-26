import numpy as np
from snlearn.message import Message
from snlearn.agent import Agent
import networkx as nx

class SocialNetwork:
    def __init__(self, num_agents, prob_in, prob_out):
        self.num_agents = num_agents
        self.prob_in = prob_in
        self.prob_out = prob_out
        self.adj_matrix = self._generate_sbm()

    def _generate_sbm(self):
        # For simplicity, split agents into two equal-sized communities
        sizes = [self.num_agents // 2, self.num_agents - self.num_agents // 2]
        probs = [[self.prob_in, self.prob_out], [self.prob_out, self.prob_in]]
        # Generate SBM adjacency matrix
        G = nx.stochastic_block_model(sizes, probs)
        adj = nx.to_numpy_array(G, dtype=int)
        return adj
    

    # Barabási–Albert Graph