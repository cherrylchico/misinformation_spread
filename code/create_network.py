"""
Visualize network sampling with different methods
Generates a network visualization showing community or degree structure
"""

import matplotlib.pyplot as plt
import networkx as nx
from snlearn.socialnetwork import SocialNetwork
from snlearn.draw_network import draw_network, print_community_statistics
import os
import argparse

def visualize_network(num_agents, 
                      network_type,
                      sampling_method=None,
                      m=2,
                      network_file='../data/facebook_combined.txt',
                      seed=42,
                      save_network=False):
    """Create and visualize a sampled network
    
    Args:
        num_agents: Number of agents to sample (required)
        network_type: 'facebook' or 'barabasi' (required)
        sampling_method: 'community', 'degree', or None (for Facebook only, default: None)
        m: Number of edges for Barabási-Albert (default: 2)
        network_file: Path to Facebook edge list file (default: '../data/facebook_combined.txt')
        seed: Random seed for reproducibility (default: 42)
        save_network: If True, saves network to data folder as pickle (default: False)
    """
    
    # Create network based on type
    print(f"Creating {network_type} network with {sampling_method or 'default'} sampling...")
    
    if network_type == 'facebook':
        network = SocialNetwork(
            num_agents=num_agents,
            seed=seed,
            facebook_params={
                'network_file': network_file,
                'sampling_method': sampling_method
            }
        )
    else:  # barabasi
        network = SocialNetwork(
            num_agents=num_agents,
            seed=seed,
            barabasi_params={
                'm': m
            }
        )
    
    # Compute group assignments
    print("Computing group assignments for visualization...")
    network.compute_group_assignments(method='auto')
    
    # Create visualization using draw_network module
    print("Generating visualization...")
    
    # Create filename based on parameters
    figures_dir = '../figures'
    filename = f'{network_type}_{sampling_method or "default"}_{num_agents}nodes.png'
    output_path = os.path.join(figures_dir, filename)
    
    # Draw the network
    fig, ax = draw_network(network, seed=seed, save_path=output_path)
    
    # Save network to data folder if requested
    if save_network:
        import pickle
        data_dir = '../data'
        os.makedirs(data_dir, exist_ok=True)
        
        # Create filename for network
        network_filename = f'{network_type}_{sampling_method or "default"}_{num_agents}nodes_seed{seed}.pkl'
        network_path = os.path.join(data_dir, network_filename)
        
        # Save network object as pickle
        with open(network_path, 'wb') as f:
            pickle.dump(network, f)
        
        print(f"Network object saved to: {network_path}")
        print(f"\nTo load this network in simulation:")
        print(f"  import pickle")
        print(f"  with open('{network_path}', 'rb') as f:")
        print(f"      network = pickle.load(f)")
    
    # Print community statistics
    print_community_statistics(network)
    
    return network


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize social network with different sampling methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Facebook with community-based sampling (100 agents)
  python visualize_network.py 100 facebook --sampling_method community
  
  # Facebook with degree-based sampling (200 agents)
  python visualize_network.py 200 facebook --sampling_method degree
  
  # Facebook without sampling (use full network or default)
  python visualize_network.py 100 facebook
  
  # Barabási-Albert network (150 agents, m=3)
  python visualize_network.py 150 barabasi --m 3
  
  # Custom Facebook network with all parameters
  python visualize_network.py 100 facebook --sampling_method community --network_file custom.txt --seed 123
"""
    )
    
    # Required positional arguments
    parser.add_argument('num_agents', type=int,
                       help='Number of agents to sample from the network')
    parser.add_argument('network_type', type=str, choices=['facebook', 'barabasi'],
                       help='Type of network to create: facebook or barabasi')
    
    # Optional arguments
    parser.add_argument('--sampling_method', type=str, choices=['community', 'degree'],
                       help='Sampling method for Facebook networks (optional, default: None)')
    
    # Network parameters
    parser.add_argument('--m', type=int, default=2,
                       help='Number of edges to attach for Barabási-Albert networks (default: 2)')
    parser.add_argument('--network_file', type=str, default='../data/facebook_combined.txt',
                       help='Path to Facebook edge list file (default: ../data/facebook_combined.txt)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--save_network', action='store_true',
                       help='Save network object to data folder as pickle file')
    
    args = parser.parse_args()
    
    visualize_network(
        num_agents=args.num_agents,
        network_type=args.network_type,
        sampling_method=args.sampling_method,
        m=args.m,
        network_file=args.network_file,
        seed=args.seed,
        save_network=args.save_network
    )
