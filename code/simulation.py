"""
Misinformation diffusion simulation in social networks
Based on the model described in the .tex files
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
from snlearn.socialnetwork import SocialNetwork
from snlearn.agent import Agent
from snlearn.message import Message
from typing import List, Dict
import json

class Simulation:
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize the simulation
        
        Args:
            config_file: path to JSON file with configurations
            **kwargs: configuration parameters (overrides config_file)
        """
        if config_file:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = kwargs if kwargs else self._default_config()
        
        # Network parameters
        self.num_agents = self.config.get('num_agents', 20)
        self.prob_in = self.config.get('prob_in', 0.6)  # Not used for BA model
        self.prob_out = self.config.get('prob_out', 0.1)  # Not used for BA model
        self.num_groups = self.config.get('num_groups', 2)
        self.ba_m = self.config.get('ba_m', 2)  # Number of edges to attach in BA model
        
        # Message parameters
        self.message_left_bias = self.config.get('message_left_bias', 2.0)
        self.message_right_bias = self.config.get('message_right_bias', 2.0)
        self.prob_truth = self.config.get('prob_truth', 0.5)
        
        # Agent parameters - high_reputation type (care more)
        self.high_rep_count = self.config.get('high_rep_count', 3)
        self.high_rep_reward = self.config.get('high_rep_reward', 1.0)
        self.high_rep_penalty = self.config.get('high_rep_penalty', 1.0)
        
        # Agent parameters - low_reputation type (don't care much)
        self.low_rep_reward = self.config.get('low_rep_reward', 0.5)
        self.low_rep_penalty = self.config.get('low_rep_penalty', 0.5)
        
        # Common agent parameters
        self.agent_left_bias = self.config.get('agent_left_bias', 2.0)
        self.agent_right_bias = self.config.get('agent_right_bias', 2.0)
        self.ave_reputation = self.config.get('ave_reputation', 0.0)
        self.variance_reputation = self.config.get('variance_reputation', 1.0)
        self.bias_strength = self.config.get('bias_strength', 0.3)
        self.forwarding_cost = self.config.get('forwarding_cost', 0.1)
        
        # Simulation parameters
        self.num_rounds = self.config.get('num_rounds', 10)
        # Initial senders will be the high_reputation agents (first high_rep_count agents)
        self.initial_senders = list(range(self.high_rep_count))
        
        # Initialize components
        self.network = SocialNetwork(
            self.num_agents, 
            self.prob_in, 
            self.prob_out, 
            self.num_groups,
            seed=self.config.get('seed', None),
            m=self.ba_m
        )
        
        self.agents = self._create_agents()
        self.messages = []
        self.history = {
            'reach': [],
            'forwarding_rate': [],
            'misinformation_contamination': [],
            'reputation_by_type': []
        }
        
    def _default_config(self):
        """Default configuration for a small and simple network"""
        return {
            'num_agents': 20,
            'prob_in': 0.6,
            'prob_out': 0.1,
            'num_groups': 2,
            'message_left_bias': 2.0,
            'message_right_bias': 2.0,
            'prob_truth': 0.5,
            'agent_left_bias': 2.0,
            'agent_right_bias': 2.0,
            'ave_reputation': 0.0,
            'variance_reputation': 1.0,
            'bias_strength': 0.3,
            'reputation_reward_strength': 0.5,
            'reputation_penalty_strength': 0.5,
            'forwarding_cost': 0.1,
            'num_rounds': 10,
            'initial_senders': [0],
            'seed': 42
        }
    
    def _create_agents(self):
        """Create all agents in the simulation with two types"""
        agents = []
        for i in range(self.num_agents):
            if i < self.high_rep_count:
                # Agents that care more about reputation (influencers)
                agent = Agent(
                    left_bias=self.agent_left_bias,
                    right_bias=self.agent_right_bias,
                    ave_reputation=self.ave_reputation,
                    variance_reputation=self.variance_reputation,
                    bias_strength=self.bias_strength,
                    reputation_reward_strength=self.high_rep_reward,
                    reputation_penalty_strength=self.high_rep_penalty,
                    forwarding_cost=self.forwarding_cost,
                    agent_type='high_reputation'
                )
            else:
                # Agents that don't care much about reputation (regular users)
                agent = Agent(
                    left_bias=self.agent_left_bias,
                    right_bias=self.agent_right_bias,
                    ave_reputation=self.ave_reputation,
                    variance_reputation=self.variance_reputation,
                    bias_strength=self.bias_strength,
                    reputation_reward_strength=self.low_rep_reward,
                    reputation_penalty_strength=self.low_rep_penalty,
                    forwarding_cost=self.forwarding_cost,
                    agent_type='low_reputation'
                )
            agents.append(agent)
        return agents
    
    def run_round(self, round_num):
        """Execute one round of the simulation"""
        # 1. Generate a new message
        message = Message(
            left_bias=self.message_left_bias,
            right_bias=self.message_right_bias,
            prob_truth=self.prob_truth
        )
        self.messages.append(message)
        
        # 2. Initialize who received the message
        received = set(self.initial_senders)
        received_from = {agent_id: [] for agent_id in self.initial_senders}
        
        # 3. Process diffusion in layers (BFS)
        # First, initial agents decide if they will forward
        for sender_id in self.initial_senders:
            sender_reputations = [self.agents[sender_id].reputation]
            self.agents[sender_id].average_utility(message, sender_reputations, store=True)
            self.agents[sender_id].decide_action(store=True)
        
        current_layer = [sid for sid in self.initial_senders 
                        if self.agents[sid].current_action == 1]
        processed = set(self.initial_senders)
        
        while current_layer:
            next_layer = []
            
            for sender_id in current_layer:
                # If decided to forward, send to neighbors
                neighbors = self.network.get_neighbors(sender_id)
                for neighbor_id in neighbors:
                    if neighbor_id not in received:
                        received.add(neighbor_id)
                        received_from[neighbor_id] = []
                    if sender_id not in received_from[neighbor_id]:
                        received_from[neighbor_id].append(sender_id)
                    if neighbor_id not in processed:
                        next_layer.append(neighbor_id)
                        processed.add(neighbor_id)
            
            # Agents in the next layer decide if they will forward
            current_layer = []
            for agent_id in next_layer:
                sender_reps = [self.agents[sid].reputation for sid in received_from[agent_id]]
                self.agents[agent_id].average_utility(message, sender_reps, store=True)
                self.agents[agent_id].decide_action(store=True)
                if self.agents[agent_id].current_action == 1:
                    current_layer.append(agent_id)
        
        # 4. (Already processed during BFS diffusion)
        
        # 5. Reveal truth and update reputations
        message.reveal_truth()
        for agent_id in received:
            self.agents[agent_id].update_reputation(message, store=True)
        
        # 6. Calculate metrics
        reach = len(received) / self.num_agents
        forwarding_count = sum(1 for agent_id in received if self.agents[agent_id].current_action == 1)
        forwarding_rate = forwarding_count / len(received) if received else 0
        
        false_forward_count = sum(
            1 for agent_id in received 
            if self.agents[agent_id].current_action == 1 and message.truth == 0
        )
        misinformation_contamination = false_forward_count / len(received) if received else 0
        
        self.history['reach'].append(reach)
        self.history['forwarding_rate'].append(forwarding_rate)
        self.history['misinformation_contamination'].append(misinformation_contamination)
        
        return {
            'message': message,
            'received': received,
            'forwarded': [aid for aid in received if self.agents[aid].current_action == 1],
            'reach': reach,
            'forwarding_rate': forwarding_rate,
            'misinformation_contamination': misinformation_contamination
        }
    
    def run(self):
        """Run the entire simulation"""
        print(f"Starting simulation with {self.num_agents} agents and {self.num_rounds} rounds...")
        results = []
        for round_num in range(self.num_rounds):
            result = self.run_round(round_num)
            results.append(result)
            print(f"Round {round_num + 1}: Reach={result['reach']:.2%}, "
                  f"Forwarding={result['forwarding_rate']:.2%}, "
                  f"Misinfo={result['misinformation_contamination']:.2%}, "
                  f"Truth={result['message'].truth}")
        return results
    
    def visualize_network(self, round_result=None, save_path=None):
        """Visualize the network and message diffusion"""
        G = self.network.graph
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Subplot 1: Network with colors by group
        ax1 = axes[0]
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_groups))
        node_colors = [colors[self.network.group_assignments[i] % len(colors)] 
                      for i in range(self.num_agents)]
        
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=300, ax=ax1, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                              arrowsize=10, ax=ax1, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax1)
        ax1.set_title('Social Network Structure\n(Colors indicate groups)', fontsize=12)
        ax1.axis('off')
        
        # Subplot 2: Message diffusion (if round_result provided)
        ax2 = axes[1]
        if round_result:
            message = round_result['message']
            received = round_result['received']
            forwarded = round_result['forwarded']
            
            # Node colors based on type and state
            node_colors_diff = []
            node_sizes = []
            for i in range(self.num_agents):
                if i < self.high_rep_count:
                    # High_reputation agents (influencers) - larger
                    base_size = 500
                    if i in forwarded:
                        node_colors_diff.append('darkred')  # Forwarded
                    elif i in received:
                        node_colors_diff.append('darkorange')  # Received but didn't forward
                    else:
                        node_colors_diff.append('lightblue')  # Didn't receive
                else:
                    # Low_reputation agents (regular) - smaller
                    base_size = 300
                    if i in forwarded:
                        node_colors_diff.append('red')  # Forwarded
                    elif i in received:
                        node_colors_diff.append('orange')  # Received but didn't forward
                    else:
                        node_colors_diff.append('lightgray')  # Didn't receive
                node_sizes.append(base_size)
            
            nx.draw_networkx_nodes(G, pos, node_color=node_colors_diff, 
                                  node_size=node_sizes, ax=ax2, alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.2, arrows=True, 
                                  arrowsize=10, ax=ax2, edge_color='gray')
            
            # Highlight diffusion edges
            if forwarded:
                diffusion_edges = []
                for sender in forwarded:
                    neighbors = self.network.get_neighbors(sender)
                    for neighbor in neighbors:
                        if neighbor in received:
                            diffusion_edges.append((sender, neighbor))
                
                if diffusion_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=diffusion_edges,
                                          edge_color='red', width=2, alpha=0.6,
                                          arrows=True, arrowsize=15, ax=ax2)
            
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
            
            truth_str = "True" if message.truth == 1 else "False"
            truth_color = 'green' if message.truth == 1 else 'red'
            ax2.set_title(f'Message Diffusion (Round)\n'
                         f'Truth: {truth_str} | '
                         f'Reach: {round_result["reach"]:.1%} | '
                         f'Forwarding Rate: {round_result["forwarding_rate"]:.1%}\n'
                         f'Blue: Influencers (care more) | Gray: Regular users',
                         fontsize=12, color=truth_color)
        else:
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                  node_size=300, ax=ax2, alpha=0.8)
            nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, 
                                  arrowsize=10, ax=ax2, edge_color='gray')
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax2)
            ax2.set_title('Social Network', fontsize=12)
        
        ax2.axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
    
    def create_diffusion_gif(self, results, save_path='diffusion_animation.gif', fps=1):
        """Create an animated GIF showing diffusion in each round"""
        G = self.network.graph
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame):
            ax.clear()
            
            if frame < len(results):
                round_result = results[frame]
                message = round_result['message']
                received = round_result['received']
                forwarded = round_result['forwarded']
                
                # Node colors based on type and state
                node_colors = []
                node_sizes = []
                for i in range(self.num_agents):
                    if i < self.high_rep_count:
                        # High_reputation agents (influencers) - larger
                        base_size = 500
                        if i in forwarded:
                            node_colors.append('darkred')  # Forwarded
                        elif i in received:
                            node_colors.append('darkorange')  # Received but didn't forward
                        else:
                            node_colors.append('lightblue')  # Didn't receive
                    else:
                        # Low_reputation agents (regular) - smaller
                        base_size = 300
                        if i in forwarded:
                            node_colors.append('red')  # Forwarded
                        elif i in received:
                            node_colors.append('orange')  # Received but didn't forward
                        else:
                            node_colors.append('lightgray')  # Didn't receive
                    
                    node_sizes.append(base_size)
                
                # Draw nodes
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                      node_size=node_sizes, ax=ax, alpha=0.8)
                
                # Draw normal edges
                nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=True, 
                                      arrowsize=8, ax=ax, edge_color='gray', width=0.5)
                
                # Highlight diffusion edges
                if forwarded:
                    diffusion_edges = []
                    for sender in forwarded:
                        neighbors = self.network.get_neighbors(sender)
                        for neighbor in neighbors:
                            if neighbor in received:
                                diffusion_edges.append((sender, neighbor))
                    
                    if diffusion_edges:
                        nx.draw_networkx_edges(G, pos, edgelist=diffusion_edges,
                                              edge_color='red', width=2.5, alpha=0.7,
                                              arrows=True, arrowsize=20, ax=ax)
                
                # Labels only for high_reputation agents
                labels = {i: str(i) for i in range(self.high_rep_count)}
                nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, 
                                       font_weight='bold', ax=ax)
                
                truth_str = "True" if message.truth == 1 else "False"
                truth_color = 'green' if message.truth == 1 else 'red'
                
                ax.set_title(f'Round {frame + 1}/{len(results)}\n'
                           f'Message: {truth_str} | '
                           f'Reach: {round_result["reach"]:.1%} | '
                           f'Forwarding: {round_result["forwarding_rate"]:.1%}\n'
                           f'Blue: Influencers (care more) | Gray: Regular users',
                           fontsize=14, fontweight='bold', color=truth_color)
                ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'Simulation Complete', 
                       ha='center', va='center', fontsize=20, fontweight='bold')
                ax.axis('off')
        
        anim = animation.FuncAnimation(fig, animate, frames=len(results), 
                                       interval=1000/fps, repeat=True)
        
        print(f"Saving GIF to {save_path}...")
        anim.save(save_path, writer='pillow', fps=fps)
        print(f"GIF saved successfully!")
        plt.close()
    
    def plot_metrics(self, save_path=None):
        """Plot metrics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        rounds = range(1, len(self.history['reach']) + 1)
        
        # Reach
        axes[0, 0].plot(rounds, self.history['reach'], marker='o', linewidth=2)
        axes[0, 0].set_title('Message Reach', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Fraction of Agents that Received')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1.1])
        
        # Forwarding rate
        axes[0, 1].plot(rounds, self.history['forwarding_rate'], marker='s', 
                       color='orange', linewidth=2)
        axes[0, 1].set_title('Forwarding Rate', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Fraction that Forwarded')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 1.1])
        
        # Misinformation contamination
        axes[1, 0].plot(rounds, self.history['misinformation_contamination'], 
                       marker='^', color='red', linewidth=2)
        axes[1, 0].set_title('Misinformation Contamination', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Fraction that Forwarded False Messages')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.1])
        
        # Average reputation
        avg_reputation = [np.mean([agent.reputation for agent in self.agents]) 
                         for _ in rounds]
        axes[1, 1].plot(rounds, avg_reputation, marker='d', color='green', linewidth=2)
        axes[1, 1].set_title('Average Agent Reputation', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Average Reputation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main function to run the simulation"""
    # Try to load from config.json, otherwise use default values
    import os
    config_file = 'config.json' if os.path.exists('config.json') else None
    
    if config_file:
        print("Loading configuration from config.json...")
        sim = Simulation(config_file=config_file)
    else:
        # Default configuration
        config = {
            'num_agents': 100,
            'prob_in': 0.6,
            'prob_out': 0.1,
            'num_groups': 2,
            'message_left_bias': 2.0,
            'message_right_bias': 2.0,
            'prob_truth': 0.8,
            'agent_left_bias': 2.0,
            'agent_right_bias': 2.0,
            'ave_reputation': 0.0,
            'variance_reputation': 1.0,
            'bias_strength': 0.3,
            'high_rep_count': 3,              # Number of influencers (2-4)
            'high_rep_reward': 1.0,            # High reward for influencers
            'high_rep_penalty': 1.0,           # High penalty for influencers
            'low_rep_reward': 0.5,             # Low reward for regular users
            'low_rep_penalty': 0.5,            # Low penalty for regular users
            'forwarding_cost': 0.1,
            'num_rounds': 10,
            'seed': 42
        }
        sim = Simulation(**config)
    
    results = sim.run()
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Show diffusion of the last round
    if results:
        sim.visualize_network(round_result=results[-1], 
                             save_path='network_diffusion.png')
    
    # Show metrics over time
    sim.plot_metrics(save_path='metrics.png')
    
    # Create animated GIF of diffusion
    print("\nCreating animated GIF of diffusion...")
    sim.create_diffusion_gif(results, save_path='diffusion_animation.gif', fps=1)
    
    print("\nSimulation complete!")
    print(f"Average reach: {np.mean(sim.history['reach']):.2%}")
    print(f"Average forwarding rate: {np.mean(sim.history['forwarding_rate']):.2%}")
    print(f"Average contamination: {np.mean(sim.history['misinformation_contamination']):.2%}")
    print(f"\nGenerated files:")
    print(f"  - network_diffusion.png")
    print(f"  - metrics.png")
    print(f"  - diffusion_animation.gif")


if __name__ == '__main__':
    main()

