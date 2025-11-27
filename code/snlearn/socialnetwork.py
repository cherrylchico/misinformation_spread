import numpy as np
from snlearn.message import Message
from snlearn.agent import Agent
import networkx as nx
import os

class SocialNetwork:
    def __init__(self, num_agents=None, prob_in=None, prob_out=None, num_groups=2, 
                 seed=None, m=2, network_type='barabasi_albert', network_file=None):
        """
        Initialize social network
        
        Args:
            num_agents: Number of agents (required for generated networks)
            prob_in: Not used, kept for compatibility
            prob_out: Not used, kept for compatibility
            num_groups: Number of groups for visualization
            seed: Random seed
            m: Number of edges to attach in Barabási-Albert model
            network_type: 'barabasi_albert' or 'facebook'
            network_file: Path to Facebook edge list file (required if network_type='facebook')
        """
        self.network_type = network_type
        self.num_groups = num_groups
        self.seed = seed
        self.m = m
        
        if network_type == 'facebook':
            if network_file is None:
                raise ValueError("network_file must be provided when network_type='facebook'")
            self.adj_matrix, self.graph, self.group_assignments, self.num_agents = \
                self._load_facebook_network(network_file, num_agents=num_agents)
        else:  # barabasi_albert
            if num_agents is None:
                raise ValueError("num_agents must be provided when network_type='barabasi_albert'")
            self.num_agents = num_agents
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
    
    def _load_facebook_network(self, network_file, num_agents=None):
        """Load network from Facebook edge list file
        
        Args:
            network_file: Path to edge list file
            num_agents: Optional. If specified and smaller than total nodes, 
                       selects a subgraph with top degree nodes
        """
        # Check if file exists
        if not os.path.exists(network_file):
            raise FileNotFoundError(f"Network file not found: {network_file}")
        
        # Load undirected graph from edge list
        G_undirected = nx.read_edgelist(network_file, nodetype=int)
        
        total_nodes = G_undirected.number_of_nodes()
        
        # If num_agents is specified and smaller than total, sample a subgraph
        if num_agents is not None and num_agents < total_nodes:
            if self.seed is not None:
                np.random.seed(self.seed)
            
            # Get top degree nodes to preserve network structure
            degrees = dict(G_undirected.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            selected_nodes = [node[0] for node in sorted_nodes[:num_agents]]
            
            # Create subgraph with selected nodes
            G_undirected = G_undirected.subgraph(selected_nodes).copy()
            print(f"Sampled {num_agents} nodes from Facebook network (total: {total_nodes})")
        
        # Convert to directed graph (each edge becomes bidirectional)
        G = nx.DiGraph()
        G.add_nodes_from(G_undirected.nodes())
        
        # Add edges: for each undirected edge, add both directions
        for edge in G_undirected.edges():
            G.add_edge(edge[0], edge[1])
            G.add_edge(edge[1], edge[0])
        
        num_agents = G.number_of_nodes()
        
        # Create adjacency matrix
        # Need to ensure nodes are 0-indexed and consecutive
        node_mapping = {old_node: new_node for new_node, old_node in enumerate(sorted(G.nodes()))}
        G_relabeled = nx.relabel_nodes(G, node_mapping)
        
        adj = nx.to_numpy_array(G_relabeled, dtype=int, nodelist=sorted(G_relabeled.nodes()))
        
        # Assign groups using community detection (like in the notebook)
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(G_undirected)
            # Map partitions to group assignments
            unique_communities = list(set(partition.values()))
            num_communities = len(unique_communities)
            
            # Limit to num_groups if there are more communities
            if num_communities > self.num_groups:
                # Sort communities by size and keep largest ones
                community_sizes = {comm: sum(1 for v in partition.values() if v == comm) 
                                 for comm in unique_communities}
                top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:self.num_groups]
                top_comm_ids = [comm[0] for comm in top_communities]
            else:
                top_comm_ids = unique_communities
            
            # Create mapping from community to group
            comm_to_group = {comm_id: i % self.num_groups for i, comm_id in enumerate(top_comm_ids)}
            
            group_assignments = [0] * num_agents
            for old_node, new_node in node_mapping.items():
                if old_node in partition:
                    comm_id = partition[old_node]
                    if comm_id in comm_to_group:
                        group_assignments[new_node] = comm_to_group[comm_id]
                    else:
                        # Assign to random group if not in top communities
                        group_assignments[new_node] = np.random.randint(0, self.num_groups)
        except ImportError:
            # Fallback: assign groups based on degree if community detection not available
            degrees = [G_relabeled.degree(node) for node in sorted(G_relabeled.nodes())]
            sorted_nodes = sorted(range(num_agents), key=lambda x: degrees[x], reverse=True)
            
            group_assignments = [0] * num_agents
            group_size = num_agents // self.num_groups
            
            for group_id in range(self.num_groups):
                start_idx = group_id * group_size
                end_idx = (group_id + 1) * group_size if group_id < self.num_groups - 1 else num_agents
                for idx in range(start_idx, end_idx):
                    if idx < len(sorted_nodes):
                        group_assignments[sorted_nodes[idx]] = group_id
        
        return adj, G_relabeled, group_assignments, num_agents
    
    def get_neighbors(self, agent_id):
        """Return the neighbors (destinations) of an agent"""
        return np.where(self.adj_matrix[agent_id] == 1)[0].tolist()
    
    def get_incoming(self, agent_id):
        """Return the agents that can send messages to this agent"""
        return np.where(self.adj_matrix[:, agent_id] == 1)[0].tolist()
    
    def get_top_degree_nodes(self, n):
        """Get top n nodes by degree (useful for selecting influencers)"""
        degrees = [(i, self.graph.degree(i)) for i in range(self.num_agents)]
        sorted_by_degree = sorted(degrees, key=lambda x: x[1], reverse=True)
        return [node[0] for node in sorted_by_degree[:n]]