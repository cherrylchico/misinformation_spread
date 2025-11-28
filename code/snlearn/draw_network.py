"""
Module for drawing and visualizing social networks
"""

import matplotlib.pyplot as plt
import networkx as nx
import os


def draw_network(network, seed=None, save_path=None, title=None, figsize=(14, 10), 
                 color_by='community', node_colors_dict=None, colormap='Set3', 
                 legend_labels=None, label_nodes=None):
    """Draw a social network with customizable node coloring
    
    Args:
        network: SocialNetwork object with graph and group_assignments
        seed: Random seed for layout reproducibility (default: use network.seed)
        save_path: Path to save the figure (default: None, don't save)
        title: Title for the plot (default: auto-generate based on network type)
        figsize: Figure size as (width, height) tuple (default: (14, 10))
        color_by: How to color nodes - 'community' (use group_assignments), 
                  'custom' (use node_colors_dict), 'action' (use node_colors_dict with action data),
                  'bias' (continuous political bias), 'reputation' (continuous reputation),
                  'diffusion' (discrete diffusion states)
                  (default: 'community')
        node_colors_dict: Dict mapping node_id -> group/value for custom coloring 
                          (required if color_by='custom', 'action', 'bias', 'reputation', or 'diffusion')
        colormap: Matplotlib colormap name (default: 'Set3')
        legend_labels: Dict mapping group_id -> label string for custom legend 
                       (default: None, auto-generate)
        label_nodes: List of node IDs to label with their numbers (default: None, no labels)
    
    Returns:
        fig, ax: Matplotlib figure and axes objects
    """
    if seed is None:
        seed = network.seed
    
    # Determine if continuous or discrete coloring
    continuous_types = ['bias', 'reputation']
    is_continuous = color_by in continuous_types
    
    # Determine coloring scheme
    if color_by == 'community':
        # Ensure group assignments are computed
        if network.group_assignments is None:
            network.compute_group_assignments(method='auto')
        color_data = {node: network.group_assignments[node] for node in network.graph.nodes()}
    elif color_by in ['custom', 'action', 'bias', 'reputation', 'diffusion']:
        if node_colors_dict is None:
            raise ValueError(f"node_colors_dict is required when color_by='{color_by}'")
        color_data = node_colors_dict
    else:
        raise ValueError(f"Invalid color_by value: {color_by}. Must be 'community', 'custom', 'action', 'bias', 'reputation', or 'diffusion'")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout for better community visualization
    # Using k=0.5 and seed to ensure consistent layout
    pos = nx.spring_layout(network.graph, k=0.5, iterations=50, seed=seed)
    
    # Handle coloring based on type
    if is_continuous:
        # Continuous colormap for bias/reputation
        cmap = plt.cm.get_cmap(colormap)
        values = [color_data[node] for node in network.graph.nodes()]
        
        # Normalize values
        if color_by == 'bias':
            # Bias is in [-1, 1], normalize to [0, 1]
            vmin, vmax = -1, 1
            norm_values = [(v + 1) / 2 for v in values]
        else:
            # Reputation or other continuous values
            vmin = min(values)
            vmax = max(values)
            v_range = vmax - vmin if vmax != vmin else 1
            norm_values = [(v - vmin) / v_range for v in values]
        
        node_colors = [cmap(norm_v) for norm_v in norm_values]
    else:
        # Discrete colormap for communities/actions/diffusion
        unique_groups = sorted(set(color_data.values()))
        cmap = plt.cm.get_cmap(colormap)
        colors = cmap(range(len(unique_groups)))
        color_map = {group: colors[i] for i, group in enumerate(unique_groups)}
        node_colors = [color_map[color_data[node]] for node in network.graph.nodes()]
    
    # Get node degrees for sizing (but don't show numbers)
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
        arrows=False,
        ax=ax
    )
    
    # Add node labels if specified
    if label_nodes is not None and len(label_nodes) > 0:
        labels = {node_id: str(node_id) for node_id in label_nodes if node_id in network.graph.nodes()}
        nx.draw_networkx_labels(network.graph, pos, labels=labels, font_size=10, 
                               font_weight='bold', ax=ax)
    
    # Add legend or colorbar
    if is_continuous:
        # Add colorbar for continuous values
        sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(colormap), 
                                   norm=plt.Normalize(vmin=vmin if color_by == 'bias' else min(values), 
                                                      vmax=vmax if color_by == 'bias' else max(values)))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        
        if color_by == 'bias':
            cbar.set_label('Political Bias', rotation=270, labelpad=15)
        elif color_by == 'reputation':
            cbar.set_label('Reputation', rotation=270, labelpad=15)
    else:
        # Add legend for discrete values
        if legend_labels is None:
            if color_by == 'community':
                legend_labels = {group: f'Community {group}' for group in unique_groups}
            elif color_by == 'action':
                # Common action labels
                action_map = {0: 'No action', 1: 'Forwarded'}
                legend_labels = {group: action_map.get(group, f'Action {group}') 
                               for group in unique_groups}
            elif color_by == 'diffusion':
                # Default diffusion labels
                diffusion_map = {0: 'Not Reached', 1: 'Received', 2: 'Forwarded'}
                legend_labels = {group: diffusion_map.get(group, f'State {group}') 
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
    
    # Add statistics (without showing node numbers in legend)
    if is_continuous:
        # For continuous values, show min/max/mean
        values_list = list(color_data.values())
        stats_text = f"Total nodes: {network.num_agents}\n"
        stats_text += f"Total edges: {network.graph.number_of_edges() // 2}\n"
        stats_text += f"Avg degree: {sum(degrees) / len(degrees):.1f}\n"
        stats_text += f"Min value: {min(values_list):.2f}\n"
        stats_text += f"Max value: {max(values_list):.2f}\n"
        stats_text += f"Mean value: {sum(values_list)/len(values_list):.2f}"
    elif color_by == 'community':
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
