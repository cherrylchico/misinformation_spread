"""
Simple script to run a simulation and plot only the reach metric over time.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(__file__))

from snlearn.simulation import Simulation
from snlearn.output_manager import OutputManager

def main():
    parser = argparse.ArgumentParser(description='Run simulation and plot only reach')
    parser.add_argument('--config-sim', type=str, required=True,
                       help='Path to simulation config JSON')
    parser.add_argument('--config-agent', type=str, required=True,
                       help='Path to agent config JSON')
    parser.add_argument('--config-influencer', type=str, default=None,
                       help='Path to influencer config JSON (optional)')
    parser.add_argument('--config-bot', type=str, default=None,
                       help='Path to bot config JSON (optional)')
    parser.add_argument('--output', type=str, default='reach_plot.png',
                       help='Output filename for the plot')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load configurations
    with open(args.config_sim, 'r') as f:
        config_sim = json.load(f)
    
    with open(args.config_agent, 'r') as f:
        config_agent = json.load(f)
    
    config_influencer = None
    if args.config_influencer and os.path.exists(args.config_influencer):
        with open(args.config_influencer, 'r') as f:
            config_influencer = json.load(f)
    
    config_bot = None
    if args.config_bot and os.path.exists(args.config_bot):
        with open(args.config_bot, 'r') as f:
            config_bot = json.load(f)
    
    # Run simulation
    print("Running simulation...")
    sim = Simulation(config_sim, config_agent, config_influencer, config_bot)
    sim.run()
    
    # Plot only reach
    plt.figure(figsize=(10, 6))
    plt.plot(sim.history['reach'], linewidth=2, color='#2E86AB')
    plt.axhline(y=np.mean(sim.history['reach']), 
                color='red', linestyle='--', linewidth=1.5,
                label=f'Mean: {np.mean(sim.history["reach"]):.2%}')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Reach (% of network)', fontsize=12)
    plt.title('Message Reach Over Time', fontsize=14, pad=15)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    
    # Save
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {args.output}")
    print(f"✓ Average reach: {np.mean(sim.history['reach']):.2%}")
    plt.close()

if __name__ == '__main__':
    main()
