import numpy as np
from snlearn.message import Message
from snlearn.agent import Agent
import networkx as nx

class SocialNetwork:
    def __init__(self, num_agents, prob_in=None, prob_out=None, num_groups=2, seed=None, m=2):
        self.num_agents = num_agents
        self.prob_in = prob_in  # Not used for BA, kept for compatibility
        self.prob_out = prob_out  # Not used for BA, kept for compatibility
        self.num_groups = num_groups
        self.seed = seed
        self.m = m  # Number of edges to attach from a new node to existing nodes
        self.adj_matrix, self.graph, self.group_assignments = self._generate_barabasi_albert()

    def _generate_barabasi_albert(self):
        """Generate a network using Barabási-Albert model"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Generate undirected Barabási-Albert graph
        # Note: BA model is undirected, so we'll convert to directed
        G_undirected = nx.barabasi_albert_graph(
            n=self.num_agents,
            m=self.m,
            seed=self.seed
        )
        
        # Convert to directed graph (each edge becomes bidirectional)
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))
        
        # Add edges: for each undirected edge, add both directions
        for edge in G_undirected.edges():
            G.add_edge(edge[0], edge[1])
            G.add_edge(edge[1], edge[0])
        
        # Create adjacency matrix
        adj = nx.to_numpy_array(G, dtype=int)
        
        # Assign groups based on node degree (for visualization purposes)
        # Higher degree nodes get different groups
        degrees = [G.degree(node) for node in range(self.num_agents)]
        sorted_nodes = sorted(range(self.num_agents), key=lambda x: degrees[x], reverse=True)
        
        group_assignments = [0] * self.num_agents
        group_size = self.num_agents // self.num_groups
        
        for group_id in range(self.num_groups):
            start_idx = group_id * group_size
            end_idx = (group_id + 1) * group_size if group_id < self.num_groups - 1 else self.num_agents
            for idx in range(start_idx, end_idx):
                if idx < len(sorted_nodes):
                    group_assignments[sorted_nodes[idx]] = group_id
        
        return adj, G, group_assignments
    
    def get_neighbors(self, agent_id):
        """Return the neighbors (destinations) of an agent"""
        return np.where(self.adj_matrix[agent_id] == 1)[0].tolist()
    
    def get_incoming(self, agent_id):
        """Return the agents that can send messages to this agent"""
        return np.where(self.adj_matrix[:, agent_id] == 1)[0].tolist()