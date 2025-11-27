import numpy as np
from snlearn.message import Message
from snlearn.agent import Agent
import networkx as nx
import os

try:
    import community as community_louvain
    COMMUNITY_AVAILABLE = True
except ImportError:
    COMMUNITY_AVAILABLE = False

class SocialNetwork:
    def __init__(self, num_agents, seed=None, barabasi_params=None, facebook_params=None):
        """
        Initialize social network
        
        Args:
            num_agents: Number of agents in the network
            seed: Random seed for reproducibility
            barabasi_params: Dict with Barabási-Albert parameters:
                - m: Number of edges to attach from a new node (default: 2)
                - num_groups: Number of groups for visualization (default: 2)
            facebook_params: Dict with Facebook network parameters:
                - network_file: Path to edge list file (required)
                - num_groups: Number of groups for visualization (default: 2)
                - sampling_method: 'community' (community-based), 'degree' (top-degree), or None (no sampling, default: None)
        
        Note: Provide either barabasi_params or facebook_params, not both.
        """
        self.num_agents = num_agents
        self.seed = seed
        
        # Determine network type based on provided parameters
        if barabasi_params is not None and facebook_params is not None:
            raise ValueError("Provide either barabasi_params or facebook_params, not both")
        
        if facebook_params is not None:
            # Facebook network
            self.network_type = 'facebook'
            if 'network_file' not in facebook_params:
                raise ValueError("facebook_params must include 'network_file'")
            
            self.network_file = facebook_params['network_file']
            self.num_groups = facebook_params.get('num_groups', 2)
            self.sampling_method = facebook_params.get('sampling_method', None)
            
            self.adj_matrix, self.graph, self.num_agents = \
                self._load_facebook_network(self.network_file, num_agents=num_agents, sampling_method=self.sampling_method)
            self.group_assignments = None
        
        elif barabasi_params is not None:
            # Barabási-Albert network
            self.network_type = 'barabasi_albert'
            self.m = barabasi_params.get('m', 2)
            self.num_groups = barabasi_params.get('num_groups', 2)
            
            self.adj_matrix, self.graph = self._generate_barabasi_albert()
            self.group_assignments = None
        
        else:
            # Default to Barabási-Albert with default parameters
            self.network_type = 'barabasi_albert'
            self.m = 2
            self.num_groups = 2
            
            self.adj_matrix, self.graph = self._generate_barabasi_albert()
            self.group_assignments = None

    def _assign_groups_by_degree(self, graph, num_agents):
        """Assign groups based on node degree for visualization
        
        Args:
            graph: NetworkX graph object
            num_agents: Number of agents in the network
            
        Returns:
            List of group assignments for each agent
        """
        # Assign groups based on node degree (for visualization purposes)
        # Higher degree nodes get different groups
        degrees = [graph.degree(node) for node in range(num_agents)]
        sorted_nodes = sorted(range(num_agents), key=lambda x: degrees[x], reverse=True)
        
        group_assignments = [0] * num_agents
        group_size = num_agents // self.num_groups
        
        for group_id in range(self.num_groups):
            start_idx = group_id * group_size
            end_idx = (group_id + 1) * group_size if group_id < self.num_groups - 1 else num_agents
            for idx in range(start_idx, end_idx):
                if idx < len(sorted_nodes):
                    group_assignments[sorted_nodes[idx]] = group_id
        
        return group_assignments

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
        
        return adj, G
    
    def _detect_communities(self, G_undirected):
        """Detect communities in the network using Louvain algorithm
        
        Args:
            G_undirected: Undirected NetworkX graph
            
        Returns:
            Tuple of (partition dict, communities dict) where:
            - partition: {node_id: community_id}
            - communities: {community_id: [node_ids]}
            Returns (None, None) if community detection fails
        """
        if not COMMUNITY_AVAILABLE:
            return None, None
        
        # Detect communities using Louvain algorithm
        partition = community_louvain.best_partition(G_undirected, random_state=self.seed)
        
        # Group nodes by community
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        
        return partition, communities
    
    def _sample_by_community(self, G_undirected, num_agents):
        """Sample nodes using community detection for better representativeness
        
        Samples a mix of high-degree (well-connected), low-degree (peripheral), 
        and bridge nodes from each community to preserve structural diversity.
        
        Args:
            G_undirected: Undirected NetworkX graph
            num_agents: Number of nodes to sample
            
        Returns:
            List of selected node IDs
        """
        # Detect communities
        partition, communities = self._detect_communities(G_undirected)
        
        if communities is None:
            # Fallback to degree-based sampling if community detection fails
            print("Warning: python-louvain not installed. Falling back to degree-based sampling.")
            degrees = dict(G_undirected.degree())
            sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
            return [node[0] for node in sorted_nodes[:num_agents]]
        
        # Sort communities by size (descending)
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Calculate nodes to sample from each community proportionally
        selected_nodes = []
        total_nodes = G_undirected.number_of_nodes()
        
        # Sample proportionally from each community with diversity
        for comm_id, nodes in sorted_communities:
            if len(selected_nodes) >= num_agents:
                break
            
            # Proportional allocation
            proportion = len(nodes) / total_nodes
            target_from_community = max(1, int(num_agents * proportion))
            
            # Don't exceed remaining quota
            target_from_community = min(target_from_community, num_agents - len(selected_nodes))
            
            # Sort nodes by degree within this community
            community_degrees = [(node, G_undirected.degree(node)) for node in nodes]
            community_degrees.sort(key=lambda x: x[1], reverse=True)
            
            # Split into high-degree (top 30%), medium (middle 40%), low-degree (bottom 30%)
            n_nodes = len(community_degrees)
            high_cutoff = int(n_nodes * 0.3)
            low_cutoff = int(n_nodes * 0.7)
            
            high_degree_nodes = [node for node, _ in community_degrees[:high_cutoff]]
            medium_degree_nodes = [node for node, _ in community_degrees[high_cutoff:low_cutoff]]
            low_degree_nodes = [node for node, _ in community_degrees[low_cutoff:]]
            
            # Allocate: 40% high-degree (bridges), 30% medium, 30% low-degree (peripheral)
            n_high = max(1, int(target_from_community * 0.4))
            n_low = max(1, int(target_from_community * 0.3))
            n_medium = target_from_community - n_high - n_low
            
            # Sample from each tier
            selected_from_comm = []
            selected_from_comm.extend(high_degree_nodes[:n_high])
            selected_from_comm.extend(medium_degree_nodes[:n_medium] if n_medium > 0 else [])
            selected_from_comm.extend(low_degree_nodes[:n_low])
            
            # If we don't have enough nodes in stratified tiers, fill from remaining
            if len(selected_from_comm) < target_from_community:
                remaining_nodes = [n for n in nodes if n not in selected_from_comm]
                selected_from_comm.extend(remaining_nodes[:target_from_community - len(selected_from_comm)])
            
            selected_nodes.extend(selected_from_comm[:target_from_community])
        
        # If we haven't reached target, add more high-degree nodes
        if len(selected_nodes) < num_agents:
            remaining = num_agents - len(selected_nodes)
            all_degrees = dict(G_undirected.degree())
            unselected = [node for node in G_undirected.nodes() if node not in selected_nodes]
            unselected_sorted = sorted(unselected, key=lambda x: all_degrees[x], reverse=True)
            selected_nodes.extend(unselected_sorted[:remaining])
        
        # Verify connectivity - ensure we have a connected subgraph
        subgraph = G_undirected.subgraph(selected_nodes)
        
        if not nx.is_connected(subgraph):
            # Get all connected components, sorted by size (largest first)
            components = sorted(nx.connected_components(subgraph), key=len, reverse=True)
            largest_cc = components[0]
            
            # Start with the largest component
            connected_nodes = set(largest_cc)
            
            # For each smaller component, find shortest bridge to connect it
            for component in components[1:]:
                if len(connected_nodes) >= num_agents:
                    break
                
                # Find the shortest path between any node in this component and the connected set
                min_path = None
                min_path_length = float('inf')
                
                for comp_node in component:
                    for connected_node in connected_nodes:
                        try:
                            path = nx.shortest_path(G_undirected, comp_node, connected_node)
                            if len(path) < min_path_length:
                                min_path = path
                                min_path_length = len(path)
                        except nx.NetworkXNoPath:
                            continue
                
                # Add bridge nodes from the path (excluding endpoints already in sets)
                if min_path:
                    for node in min_path:
                        if node not in connected_nodes and len(connected_nodes) < num_agents:
                            connected_nodes.add(node)
                    # Add nodes from this component
                    for node in component:
                        if len(connected_nodes) < num_agents:
                            connected_nodes.add(node)
            
            # If we still need more nodes and added bridges, fill from high-degree unselected nodes
            if len(connected_nodes) < num_agents:
                all_degrees = dict(G_undirected.degree())
                remaining = [n for n in G_undirected.nodes() if n not in connected_nodes]
                remaining_sorted = sorted(remaining, key=lambda x: all_degrees[x], reverse=True)
                for node in remaining_sorted:
                    if len(connected_nodes) >= num_agents:
                        break
                    # Only add if it connects to existing network
                    if any(G_undirected.has_edge(node, n) for n in connected_nodes):
                        connected_nodes.add(node)
            
            selected_nodes = list(connected_nodes)[:num_agents]
            
            # Verify final connectivity
            final_subgraph = G_undirected.subgraph(selected_nodes)
            if not nx.is_connected(final_subgraph):
                # Last resort: just take largest connected component
                print(f"Warning: Could not create fully connected sample. Using largest component.")
                largest_component = max(nx.connected_components(final_subgraph), key=len)
                selected_nodes = list(largest_component)
        
        return selected_nodes
    
    def _load_facebook_network(self, network_file, num_agents=None, sampling_method=None):
        """Load network from Facebook edge list file
        
        Args:
            network_file: Path to edge list file
            num_agents: Optional. If specified and smaller than total nodes, 
                       selects a subgraph based on sampling_method
            sampling_method: 'community' (community-based), 'degree' (top-degree), or None (no sampling)
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
            
            if sampling_method == 'community':
                # Community-based sampling for better representativeness
                selected_nodes = self._sample_by_community(G_undirected, num_agents)
                print(f"Sampled {num_agents} nodes using community-based sampling from Facebook network (total: {total_nodes})")
            elif sampling_method == 'degree':
                # Top degree sampling (explicit method)
                degrees = dict(G_undirected.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                selected_nodes = [node[0] for node in sorted_nodes[:num_agents]]
                print(f"Sampled {num_agents} nodes using degree-based sampling from Facebook network (total: {total_nodes})")
            else:
                # No sampling specified, default to degree-based for backward compatibility
                degrees = dict(G_undirected.degree())
                sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
                selected_nodes = [node[0] for node in sorted_nodes[:num_agents]]
                print(f"Sampled {num_agents} nodes from Facebook network (total: {total_nodes})")
            
            # Create subgraph with selected nodes
            G_undirected = G_undirected.subgraph(selected_nodes).copy()
        
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
        
        return adj, G_relabeled, num_agents
    
    def compute_group_assignments(self, method='auto'):
        """Compute and store group assignments for visualization
        
        Args:
            method: 'auto' (community detection for Facebook, degree for Barabási-Albert),
                    'degree' (degree-based), or 'community' (community detection)
        
        Returns:
            List of group assignments for each agent
        """
        if method == 'auto':
            if self.network_type == 'facebook':
                method = 'community'
            else:
                method = 'degree'
        
        if method == 'community':
            # Get undirected version of graph for community detection
            G_undirected = self.graph.to_undirected()
            partition, communities = self._detect_communities(G_undirected)
            
            if partition is None:
                # Fallback to degree-based if community detection not available
                group_assignments = self._assign_groups_by_degree(self.graph, self.num_agents)
            else:
                # Map partitions to group assignments
                unique_communities = list(set(partition.values()))
                num_communities = len(unique_communities)
                
                # Limit to num_groups if there are more communities
                if num_communities > self.num_groups:
                    # Sort communities by size and keep largest ones
                    community_sizes = {comm: len(communities[comm]) for comm in unique_communities}
                    top_communities = sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:self.num_groups]
                    top_comm_ids = [comm[0] for comm in top_communities]
                else:
                    top_comm_ids = unique_communities
                
                # Create mapping from community to group
                comm_to_group = {comm_id: i % self.num_groups for i, comm_id in enumerate(top_comm_ids)}
                
                group_assignments = [0] * self.num_agents
                for node in range(self.num_agents):
                    if node in partition:
                        comm_id = partition[node]
                        if comm_id in comm_to_group:
                            group_assignments[node] = comm_to_group[comm_id]
                        else:
                            # Assign to random group if not in top communities
                            group_assignments[node] = np.random.randint(0, self.num_groups)
        else:  # method == 'degree'
            group_assignments = self._assign_groups_by_degree(self.graph, self.num_agents)
        
        self.group_assignments = group_assignments
        return group_assignments
    
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