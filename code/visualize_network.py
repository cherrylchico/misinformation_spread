"""
Visualize network sampling with different methods
Generates a network visualization showing community or degree structure
"""

import matplotlib.pyplot as plt
import networkx as nx
from snlearn.socialnetwork import SocialNetwork
import os
import argparse

def visualize_network(num_agents, 
                      network_type,
                      sampling_method=None,
                      m=2,
                      network_file='../data/facebook_combined.txt',
                      num_groups=5,
                      seed=42,
                      save_network=False):
    """Create and visualize a sampled network
    
    Args:
        num_agents: Number of agents to sample (required)
        network_type: 'facebook' or 'barabasi' (required)
        sampling_method: 'community', 'degree', or None (for Facebook only, default: None)
        m: Number of edges for Barabási-Albert (default: 2)
        network_file: Path to Facebook edge list file (default: '../data/facebook_combined.txt')
        num_groups: Number of groups for visualization (default: 5)
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
                'num_groups': num_groups,
                'sampling_method': sampling_method
            }
        )
    else:  # barabasi
        network = SocialNetwork(
            num_agents=num_agents,
            seed=seed,
            barabasi_params={
                'm': m,
                'num_groups': num_groups
            }
        )
    
    # Compute group assignments
    print("Computing group assignments for visualization...")
    network.compute_group_assignments(method='auto')
    
    # Create visualization
    print("Generating visualization...")
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Use spring layout for better community visualization
    pos = nx.spring_layout(network.graph, k=0.5, iterations=50, seed=seed)
    
    # Get unique groups and create color map
    unique_groups = list(set(network.group_assignments))
    colors = plt.cm.Set3(range(len(unique_groups)))
    color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
    
    # Color nodes by community
    node_colors = [color_map[network.group_assignments[node]] for node in range(network.num_agents)]
    
    # Get node degrees for sizing
    degrees = [network.graph.degree(node) for node in range(network.num_agents)]
    max_degree = max(degrees)
    min_degree = min(degrees)
    
    # Normalize sizes between 100 and 1000
    node_sizes = [100 + 900 * (deg - min_degree) / (max_degree - min_degree) if max_degree > min_degree else 300 
                  for deg in degrees]
    
    # Draw network
    nx.draw_networkx_nodes(
        network.graph, pos,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.8,
        ax=ax
    )
    
    nx.draw_networkx_edges(
        network.graph, pos,
        alpha=0.2,
        width=0.5,
        ax=ax
    )
    
    # Add legend for communities
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color_map[group], 
                                  markersize=10, label=f'Community {group}')
                      for group in unique_groups]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Title and styling
    title = f'{network_type.capitalize()} Network'
    if network_type == 'facebook':
        if sampling_method == 'community':
            title += ' - Community-Based Sampling'
        elif sampling_method == 'degree':
            title += ' - Degree-Based Sampling'
        else:
            title += ' - Default Sampling'
    
    ax.set_title(f'{title}\n{network.num_agents} Nodes', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add statistics
    stats_text = f"Total nodes: {network.num_agents}\n"
    stats_text += f"Total edges: {network.graph.number_of_edges() // 2}\n"  # Divide by 2 since it's bidirectional
    stats_text += f"Communities detected: {len(unique_groups)}\n"
    stats_text += f"Avg degree: {sum(degrees) / len(degrees):.1f}\n"
    stats_text += f"Max degree: {max_degree}\n"
    stats_text += f"Min degree: {min_degree}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Save figure
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Create filename based on parameters
    filename = f'{network_type}_{sampling_method or "default"}_{num_agents}nodes.png'
    output_path = os.path.join(figures_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_path}")
    
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
    print("\nCommunity Statistics:")
    for group in sorted(unique_groups):
        count = network.group_assignments.count(group)
        avg_deg = sum(degrees[i] for i in range(network.num_agents) 
                     if network.group_assignments[i] == group) / count
        print(f"  Community {group}: {count} nodes, avg degree: {avg_deg:.1f}")
    
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
    parser.add_argument('--num_groups', type=int, default=5,
                       help='Number of groups for visualization (default: 5)')
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
        num_groups=args.num_groups,
        seed=args.seed,
        save_network=args.save_network
    )
