"""
Example: Drawing a network colored by agent actions (e.g., who forwarded messages)
"""

from snlearn.socialnetwork import SocialNetwork
from snlearn.draw_network import draw_network
import random

# Create a network
network = SocialNetwork(
    num_agents=100,
    seed=42,
    facebook_params={
        'network_file': '../data/facebook_combined.txt',
        'sampling_method': 'community'
    }
)

# Simulate some agents forwarding messages (0 = no action, 1 = forwarded)
# In a real simulation, this would come from the actual simulation results
random.seed(42)
agent_actions = {node: random.choice([0, 1]) for node in network.graph.nodes()}

# Draw network colored by community (default)
print("Drawing network colored by community...")
draw_network(
    network, 
    save_path='../figures/example_community_coloring.png',
    title='Network Colored by Community'
)

# Draw network colored by agent actions
print("\nDrawing network colored by agent actions...")
draw_network(
    network,
    color_by='action',
    node_colors_dict=agent_actions,
    save_path='../figures/example_action_coloring.png',
    title='Network Colored by Agent Actions',
    colormap='RdYlGn',  # Red-Yellow-Green colormap
    legend_labels={0: 'Did not forward', 1: 'Forwarded message'}
)

print("\nBoth visualizations saved!")
print("- Community-based: figures/example_community_coloring.png")
print("- Action-based: figures/example_action_coloring.png")
