"""
Module for drawing and visualizing social networks
"""

import matplotlib.pyplot as plt
import networkx as nx
import os


def draw_network(network, seed=None, save_path=None, title=None, figsize=(14, 10), 
                 color_by='community', node_colors_dict=None, colormap='Set3', 
                 legend_labels=None):
    """Draw a social network with customizable node coloring
    
    Args:
        network: SocialNetwork object with graph and group_assignments
        seed: Random seed for layout reproducibility (default: use network.seed)
        save_path: Path to save the figure (default: None, don't save)
        title: Title for the plot (default: auto-generate based on network type)
        figsize: Figure size as (width, height) tuple (default: (14, 10))
        color_by: How to color nodes - 'community' (use group_assignments), 
                  'custom' (use node_colors_dict), or 'action' (use node_colors_dict with action data)
                  (default: 'community')
        node_colors_dict: Dict mapping node_id -> group/value for custom coloring 
                          (required if color_by='custom' or 'action')
        colormap: Matplotlib colormap name (default: 'Set3')
        legend_labels: Dict mapping group_id -> label string for custom legend 
                       (default: None, auto-generate)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    if seed is None:
        seed = network.seed
    
    # Determine coloring scheme
    if color_by == 'community':
        # Ensure group assignments are computed
        if network.group_assignments is None:
            network.compute_group_assignments(method='auto')
        color_data = {node: network.group_assignments[node] for node in network.graph.nodes()}
    elif color_by in ['custom', 'action']:
        if node_colors_dict is None:
            raise ValueError(f"node_colors_dict is required when color_by='{color_by}'")
        color_data = node_colors_dict
    else:
        raise ValueError(f"Invalid color_by value: {color_by}. Must be 'community', 'custom', or 'action'")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout for better community visualization
    # Using k=0.5 and seed to ensure consistent layout
    pos = nx.spring_layout(network.graph, k=0.5, iterations=50, seed=seed)
    
    # Get unique groups and create color map
    unique_groups = sorted(set(color_data.values()))
    colors = plt.cm.get_cmap(colormap)(range(len(unique_groups)))
    color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
    
    # Color nodes - iterate through actual graph nodes
    node_colors = [color_map[color_data[node]] for node in network.graph.nodes()]
    
    # Get node degrees for sizing
    degrees = [network.graph.degree(node) for node in network.graph.nodes()]
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
    
    # Add legend
    if legend_labels is None:
        if color_by == 'community':
            legend_labels = {group: f'Community {group}' for group in unique_groups}
        elif color_by == 'action':
            # Common action labels
            action_map = {0: 'No action', 1: 'Forwarded'}
            legend_labels = {group: action_map.get(group, f'Action {group}') 
                           for group in unique_groups}
        else:
            legend_labels = {group: f'Group {group}' for group in unique_groups}
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=color_map[group], 
                                  markersize=10, label=legend_labels.get(group, f'Group {group}'))
                      for group in unique_groups]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # Generate title if not provided
    if title is None:
        title = f'{network.network_type.capitalize()} Network'
        if hasattr(network, 'sampling_method') and network.sampling_method:
            if network.sampling_method == 'community':
                title += ' - Community-Based Sampling'
            elif network.sampling_method == 'degree':
                title += ' - Degree-Based Sampling'
    
    ax.set_title(f'{title}\n{network.num_agents} Nodes', 
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    # Add statistics
    if color_by == 'community':
        stats_text = f"Total nodes: {network.num_agents}\n"
        stats_text += f"Total edges: {network.graph.number_of_edges() // 2}\n"
        stats_text += f"Communities detected: {len(unique_groups)}\n"
        stats_text += f"Avg degree: {sum(degrees) / len(degrees):.1f}\n"
        stats_text += f"Max degree: {max_degree}\n"
        stats_text += f"Min degree: {min_degree}"
    else:
        # For action-based or custom coloring, show distribution
        stats_text = f"Total nodes: {network.num_agents}\n"
        stats_text += f"Total edges: {network.graph.number_of_edges() // 2}\n"
        stats_text += f"Avg degree: {sum(degrees) / len(degrees):.1f}\n"
        for group in unique_groups:
            count = sum(1 for v in color_data.values() if v == group)
            label = legend_labels.get(group, f'Group {group}')
            stats_text += f"{label}: {count} nodes\n"
        # Remove trailing newline
        stats_text = stats_text.rstrip()
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    return fig, ax


def print_community_statistics(network):
    """Print statistics for each community in the network
    
    Args:
        network: SocialNetwork object with group_assignments
    """
    if network.group_assignments is None:
        network.compute_group_assignments(method='auto')
    
    unique_groups = sorted(set(network.group_assignments))
    degrees = [network.graph.degree(node) for node in network.graph.nodes()]
    
    print("\nCommunity Statistics:")
    for group in unique_groups:
        count = network.group_assignments.count(group)
        avg_deg = sum(degrees[i] for i in range(network.num_agents) 
                     if network.group_assignments[i] == group) / count
        print(f"  Community {group}: {count} nodes, avg degree: {avg_deg:.1f}")
