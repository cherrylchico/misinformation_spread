import numpy as np
from snlearn.message import Message
from snlearn.agent import Agent
import networkx as nx

class SocialNetwork:
    def __init__(self, num_agents, prob_in, prob_out, num_groups=2, seed=None):
        self.num_agents = num_agents
        self.prob_in = prob_in
        self.prob_out = prob_out
        self.num_groups = num_groups
        self.seed = seed
        self.adj_matrix, self.graph, self.group_assignments = self._generate_sbm()

    def _generate_sbm(self):
        """Gera uma rede usando Stochastic Block Model"""
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Divide agentes em grupos
        sizes = []
        remaining = self.num_agents
        for i in range(self.num_groups - 1):
            size = remaining // (self.num_groups - i)
            sizes.append(size)
            remaining -= size
        sizes.append(remaining)
        
        # Cria matriz de probabilidades
        probs = np.full((self.num_groups, self.num_groups), self.prob_out)
        np.fill_diagonal(probs, self.prob_in)
        
        # Gera grafo manualmente (mais compatível)
        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))
        
        # Atribui grupos aos agentes
        group_assignments = []
        node_idx = 0
        for group_id, size in enumerate(sizes):
            for _ in range(size):
                group_assignments.append(group_id)
                node_idx += 1
        
        # Adiciona arestas baseado nas probabilidades
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i != j:
                    group_i = group_assignments[i]
                    group_j = group_assignments[j]
                    prob = probs[group_i, group_j]
                    if np.random.random() < prob:
                        G.add_edge(i, j)
        
        adj = nx.to_numpy_array(G, dtype=int)
        
        return adj, G, group_assignments
    
    def get_neighbors(self, agent_id):
        """Retorna os vizinhos (destinos) de um agente"""
        return np.where(self.adj_matrix[agent_id] == 1)[0].tolist()
    
    def get_incoming(self, agent_id):
        """Retorna os agentes que podem enviar mensagens para este agente"""
        return np.where(self.adj_matrix[:, agent_id] == 1)[0].tolist()