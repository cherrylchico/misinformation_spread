"""
Misinformation diffusion simulation in social networks
Main script to run simulations
"""

import numpy as np
import os
import argparse
from snlearn.simulation import Simulation


def main():
    """Main function to run the simulation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run misinformation diffusion simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config.json
  python run_simulation.py
  
  # Run with specific config file
  python run_simulation.py --config config/config_generate_network.json
  
  # Run with Facebook config
  python run_simulation.py --config config/config_facebook.json
"""
    )
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Path to configuration JSON file (default: config/config.json)')
    
    args = parser.parse_args()
    
    # Try to load from specified config file, otherwise use default values
    config_file = args.config if os.path.exists(args.config) else None
    
    if config_file:
        print(f"Loading configuration from {config_file}...")
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
    
    # Create figures directory if it doesn't exist
    figures_dir = '../figures'
    os.makedirs(figures_dir, exist_ok=True)
    
    # Show diffusion of the last round
    if results:
        sim.visualize_network(round_result=results[-1], 
                             save_path=os.path.join(figures_dir, 'network_diffusion.png'))
    
    # Show metrics over time
    sim.plot_metrics(save_path=os.path.join(figures_dir, 'metrics.png'))
    
    # Create animated GIF of diffusion
    print("\nCreating animated GIF of diffusion...")
    sim.create_diffusion_gif(results, save_path=os.path.join(figures_dir, 'diffusion_animation.gif'), fps=1)
    
    print("\nSimulation complete!")
    print(f"Average reach: {np.mean(sim.history['reach']):.2%}")
    print(f"Average forwarding rate: {np.mean(sim.history['forwarding_rate']):.2%}")
    print(f"Average contamination: {np.mean(sim.history['misinformation_contamination']):.2%}")
    print(f"\nGenerated files in {figures_dir}:")
    print(f"  - network_diffusion.png")
    print(f"  - metrics.png")
    print(f"  - diffusion_animation.gif")


if __name__ == '__main__':
    main()

