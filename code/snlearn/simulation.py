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
import pickle
import os

class Simulation:
    def __init__(self, config_sim, config_agent, config_influencer=None, config_bot=None):
        """
        Initialize the simulation
        
        Args:
            config_sim: dict or path to JSON file with simulation-level configurations
                       Required keys: 'network_pickle_file', 'num_rounds'
                       Optional keys: 'message' (dict with left_bias, right_bias, prob_truth),
                                     'num_initial_senders' (number of initial senders, default=3)
                       
            config_agent: dict with agent configuration (used for regular agents)
                         Required keys: 'count' (None = all remaining agents)
                         Optional keys: all agent parameters (left_bias, right_bias, ave_reputation, etc.)
                         
            config_influencer: dict with influencer configuration (optional)
                              If None, no influencers in simulation
                              Required keys: 'count'
                              Optional keys: all agent parameters
                              
            config_bot: dict with bot configuration (optional)
                       If None, no bots in simulation
                       Required keys: 'count'
                       Optional keys: all agent parameters
        
        Example:
            config_sim = {
                'network_pickle_file': 'networks/facebook_network.pkl',
                'num_rounds': 10,
                'num_initial_senders': 3,
                'message': {'left_bias': 0.5, 'right_bias': 0.5, 'prob_truth': 0.5}
            }
            
            config_agent = {
                'count': None,
                'ave_reputation': 0.0,
                'variance_reputation': 1.0,
                'bias_strength': 0.5,
                'reputation_reward_strength': 0.5,
                'reputation_penalty_strength': 0.5,
                'forwarding_cost': 0.1
            }
            
            config_influencer = {
                'count': 3,
                'ave_reputation': 1.0,
                'variance_reputation': 1.0,
                'bias_strength': 1.0,
                'reputation_reward_strength': 1.0,
                'reputation_penalty_strength': 1.0,
                'forwarding_cost': 0.1
            }
            
            sim = Simulation(config_sim, config_agent, config_influencer)
        """
        # Load simulation config from file or use dict
        if isinstance(config_sim, str):
            with open(config_sim, 'r') as f:
                self.config_sim = json.load(f)
        else:
            self.config_sim = config_sim
        
        # Store agent configurations
        self.regular_config = config_agent if config_agent else self._default_regular_config()
        self.influencer_config = config_influencer
        self.bot_config = config_bot
        
        # Network parameters
        self.network_pickle_file = self.config_sim.get('network_pickle_file', None)
        
        # Message parameters
        message_config = self.config_sim.get('message', {})
        self.message_left_bias = message_config.get('left_bias', 0.5)
        self.message_right_bias = message_config.get('right_bias', 0.5)
        self.prob_truth = message_config.get('prob_truth', 0.5)
        
        # Simulation parameters
        self.num_rounds = self.config_sim.get('num_rounds', 10)
        self.num_initial_senders = self.config_sim.get('num_initial_senders', 3)
        
        # Load network from pickle file
        if self.network_pickle_file is None:
            raise ValueError("network_pickle_file is required. Use create_network.py to generate a network first.")
        

        if not os.path.exists(self.network_pickle_file):
            raise FileNotFoundError(f"Network pickle file not found: {self.network_pickle_file}")
        
        print(f"Loading network from pickle file: {self.network_pickle_file}")
        with open(self.network_pickle_file, 'rb') as f:
            self.network = pickle.load(f)
        
        # Update parameters from loaded network
        self.num_agents = self.network.num_agents
        self.network_type = self.network.network_type
        print(f"Loaded {self.network_type} network with {self.num_agents} agents")
        
        # Determine agent counts
        num_influencers = self.influencer_config.get('count', 0) if self.influencer_config else 0
        num_bots = self.bot_config.get('count', 0) if self.bot_config else 0
        num_regular = self.regular_config.get('count', None)
        
        # If num_regular is None, it takes remaining agents
        if num_regular is None:
            num_regular = self.num_agents - num_influencers - num_bots
        
        # Validate distribution
        if num_influencers + num_regular + num_bots != self.num_agents:
            raise ValueError(f"Agent distribution [influencers={num_influencers}, regular={num_regular}, bots={num_bots}] "
                           f"does not sum to total agents ({self.num_agents})")
        
        # Store counts
        self.num_influencers = num_influencers
        self.num_regular = num_regular
        self.num_bots = num_bots
        
        # Determine influencers - use top nodes by degree
        if self.num_influencers > 0:
            self.influencers = self.network.get_top_degree_nodes(self.num_influencers)
        else:
            self.influencers = []
        
        # Initial senders are always the top degree nodes, regardless of influencer status
        self.initial_senders = self.network.get_top_degree_nodes(self.num_initial_senders)
        
        print(f"Agent distribution: {self.num_influencers} influencers, {self.num_regular} regular, {self.num_bots} bots")
        
        self.agents = self._create_agents()
        self.messages = []
        self.history = {
            'reach': [],
            'forwarding_rate': [],
            'misinformation_contamination': [],
            'average_reputation': [],
            'reputation_by_type': []
        }
    
    def _default_regular_config(self):
        """Default configuration for regular agents"""
        return {
            'count': None,
            'left_bias': 0.5,
            'right_bias': 0.5,
            'ave_reputation': 0.0,
            'variance_reputation': 1.0,
            'bias_strength': 0.5,
            'reputation_reward_strength': 0.5,
            'reputation_penalty_strength': 0.5,
            'forwarding_cost': 0.1
        }
    
    def _create_agents(self):
        """Create all agents in the simulation with three types based on distribution"""
        agents = []
        influencer_set = set(self.influencers)
        
        # Determine which nodes are bots (last num_bots nodes)
        bot_nodes = set(range(self.num_agents - self.num_bots, self.num_agents))
        
        for i in range(self.num_agents):
            if i in influencer_set and self.influencer_config:
                # Influencer agents
                agent = Agent(
                    left_bias=self.influencer_config.get('left_bias', 0.5),
                    right_bias=self.influencer_config.get('right_bias', 0.5),
                    ave_reputation=self.influencer_config.get('ave_reputation', 1.0),
                    variance_reputation=self.influencer_config.get('variance_reputation', 1.0),
                    bias_strength=self.influencer_config.get('bias_strength', 1.0),
                    reputation_reward_strength=self.influencer_config.get('reputation_reward_strength', 1.0),
                    reputation_penalty_strength=self.influencer_config.get('reputation_penalty_strength', 1.0),
                    forwarding_cost=self.influencer_config.get('forwarding_cost', 0.1),
                    agent_type='high_reputation',
                    type='influencer'
                )
            elif i in bot_nodes and self.bot_config:
                # Bot agents
                agent = Agent(
                    left_bias=self.bot_config.get('left_bias', 0.5),
                    right_bias=self.bot_config.get('right_bias', 0.5),
                    ave_reputation=self.bot_config.get('ave_reputation', 0.0),
                    variance_reputation=self.bot_config.get('variance_reputation', 1.0),
                    bias_strength=self.bot_config.get('bias_strength', 1.0),
                    reputation_reward_strength=self.bot_config.get('reputation_reward_strength', 0.0),
                    reputation_penalty_strength=self.bot_config.get('reputation_penalty_strength', 0.0),
                    forwarding_cost=self.bot_config.get('forwarding_cost', 0.1),
                    agent_type='bot',
                    type='bot'
                )
            else:
                # Regular agents
                agent = Agent(
                    left_bias=self.regular_config.get('left_bias', 0.5),
                    right_bias=self.regular_config.get('right_bias', 0.5),
                    ave_reputation=self.regular_config.get('ave_reputation', 0.0),
                    variance_reputation=self.regular_config.get('variance_reputation', 1.0),
                    bias_strength=self.regular_config.get('bias_strength', 0.5),
                    reputation_reward_strength=self.regular_config.get('reputation_reward_strength', 0.5),
                    reputation_penalty_strength=self.regular_config.get('reputation_penalty_strength', 0.5),
                    forwarding_cost=self.regular_config.get('forwarding_cost', 0.1),
                    agent_type='low_reputation',
                    type='regular'
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
        
        # Store average reputation after this round
        avg_reputation = np.mean([agent.reputation for agent in self.agents])
        self.history['average_reputation'].append(avg_reputation)
        
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
        
        # Track initial reputation (round 0)
        initial_reputation = np.mean([agent.reputation for agent in self.agents])
        print(f"Round 0: Initial Average Reputation={initial_reputation:.4f}")
        
        results = []
        for round_num in range(self.num_rounds):
            result = self.run_round(round_num)
            results.append(result)
            
            # Calculate average reputation after this round
            avg_reputation = np.mean([agent.reputation for agent in self.agents])
            
            #             print(f"Round {round_num + 1}: Reach={result['reach']:.2%}, "
            #                   f"Forwarding={result['forwarding_rate']:.2%}, "
            #                   f"Misinfo={result['misinformation_contamination']:.2%}, "
            #                   f"Truth={result['message'].truth}, "
            #                   f"Avg Reputation={avg_reputation:.4f}")
        return results
    
    def visualize_network(self, round_result=None, save_path=None):
        """Visualize the network and message diffusion using draw_network"""
        from .draw_network import draw_network
        import os
        
        # Determine base path for saving
        if save_path:
            base_path = os.path.splitext(save_path)[0]
        else:
            base_path = None
        
        # 1. Political Bias visualization
        bias_colors = {i: self.agents[i].bias for i in range(self.num_agents)}
        bias_save_path = f"{base_path}_political_bias.png" if base_path else None
        draw_network(
            self.network,
            seed=self.network.seed,
            save_path=bias_save_path,
            title='Political Bias (Blue=Left, Red=Right)',
            figsize=(14, 10),
            color_by='bias',
            node_colors_dict=bias_colors,
            colormap='RdBu_r',
            label_nodes=self.initial_senders
        )
        plt.close()
        
        # 2. Initial Reputation visualization
        reputation_colors = {i: self.agents[i].baseline_reputation for i in range(self.num_agents)}
        rep_save_path = f"{base_path}_initial_reputation.png" if base_path else None
        draw_network(
            self.network,
            seed=self.network.seed,
            save_path=rep_save_path,
            title='Initial Reputation (Purple=Low, Yellow=High)',
            figsize=(14, 10),
            color_by='reputation',
            node_colors_dict=reputation_colors,
            colormap='viridis',
            label_nodes=self.initial_senders
        )
        plt.close()
        
        # 3. Message Diffusion visualization (if round_result provided)
        if round_result:
            message = round_result['message']
            received = round_result['received']
            forwarded = round_result['forwarded']
            
            # Map agents to diffusion state: 0=not reached, 1=received, 2=forwarded
            diffusion_colors = {}
            for i in range(self.num_agents):
                if i in forwarded:
                    diffusion_colors[i] = 2  # Forwarded
                elif i in received:
                    diffusion_colors[i] = 1  # Received but didn't forward
                else:
                    diffusion_colors[i] = 0  # Didn't receive
            
            truth_str = "True" if message.truth == 1 else "False"
            diffusion_title = f'Message Diffusion - Truth: {truth_str}'
            diffusion_save_path = f"{base_path}_message_diffusion.png" if base_path else None
            
            legend_labels = {
                0: 'Not Reached',
                1: 'Received, Not Forwarded',
                2: 'Forwarded'
            }
            
            draw_network(
                self.network,
                seed=self.network.seed,
                save_path=diffusion_save_path,
                title=diffusion_title,
                figsize=(14, 10),
                color_by='diffusion',
                node_colors_dict=diffusion_colors,
                colormap='RdYlGn',
                legend_labels=legend_labels,
                label_nodes=self.initial_senders
            )
            plt.close()
    
    def create_diffusion_gif(self, results, save_path='diffusion_animation.gif', fps=1):
        """Create an animated GIF showing diffusion in each round"""
        G = self.network.graph
        # Use network's seed for consistent layout
        pos = nx.spring_layout(G, seed=self.network.seed, k=0.5, iterations=50)
        
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
                influencer_set = set(self.influencers)
                
                for i in range(self.num_agents):
                    if i in influencer_set:
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
                labels = {i: str(i) for i in self.influencers}
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
        """Plot metrics over time and always save to file"""
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
        axes[1, 1].plot(rounds, self.history['average_reputation'], marker='d', color='green', linewidth=2)
        axes[1, 1].set_title('Average Agent Reputation', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Round')
        axes[1, 1].set_ylabel('Average Reputation')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # Always save, use default filename if not provided
        if save_path is None:
            save_path = "simulation_metrics.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()