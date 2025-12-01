"""
Misinformation diffusion simulation in social networks
Main script to run simulations
"""

import numpy as np
import os
import argparse
import json
from snlearn.simulation import Simulation
from snlearn.output_manager import OutputManager


def main():
    """Main function to run the simulation"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Run misinformation diffusion simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all agent types (influencers, regular, bots)
  python run_simulation.py --config-sim config/config_sim_general.json \\
                          --config-agent config/config_sim_agent.json \\
                          --config-influencer config/config_sim_influencer.json \\
                          --config-bot config/config_sim_bot.json
  
  # Run with only regular agents
  python run_simulation.py --config-sim config/config_sim_general.json \\
                          --config-agent config/config_sim_agent.json
  
  # Run with influencers but no bots
  python run_simulation.py --config-sim config/config_sim_general.json \\
                          --config-agent config/config_sim_agent.json \\
                          --config-influencer config/config_sim_influencer.json
"""
    )
    parser.add_argument('--config-sim', type=str, required=True,
                       help='Path to simulation config JSON file (required)')
    parser.add_argument('--config-agent', type=str, required=True,
                       help='Path to agent config JSON file (required)')
    parser.add_argument('--config-influencer', type=str, default=None,
                       help='Path to influencer config JSON file (optional)')
    parser.add_argument('--config-bot', type=str, default=None,
                       help='Path to bot config JSON file (optional)')
    parser.add_argument('--no-gif', action='store_true',
                       help='Skip GIF generation (saves time for quick runs)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility (optional, default: random)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to: {args.seed}")
    
    # Load simulation config (can be path or will be loaded in Simulation class)
    config_sim = args.config_sim
    
    # Load agent config
    print(f"Loading agent configuration from {args.config_agent}...")
    with open(args.config_agent, 'r') as f:
        config_agent = json.load(f)
    
    # Load optional influencer config
    config_influencer = None
    if args.config_influencer and os.path.exists(args.config_influencer):
        print(f"Loading influencer configuration from {args.config_influencer}...")
        with open(args.config_influencer, 'r') as f:
            config_influencer = json.load(f)
    
    # Load optional bot config
    config_bot = None
    if args.config_bot and os.path.exists(args.config_bot):
        print(f"Loading bot configuration from {args.config_bot}...")
        with open(args.config_bot, 'r') as f:
            config_bot = json.load(f)
    
    # Initialize simulation
    print("\nInitializing simulation...")
    sim = Simulation(config_sim, config_agent, config_influencer, config_bot)
    
    # Run simulation
    print("\nRunning simulation...")
    results = sim.run()
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    # Create timestamped output directory
    output_dir = OutputManager.create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # Show diffusion of the last round
    if results:
        sim.visualize_network(round_result=results[-1], 
                             save_path=os.path.join(output_dir, 'network_diffusion.png'))
    
    # Show metrics over time
    sim.plot_metrics(save_path=os.path.join(output_dir, 'metrics.png'))
    
    # Create animated GIF of diffusion (optional)
    if not args.no_gif:
        print("\nCreating animated GIF of diffusion...")
        sim.create_diffusion_gif(results, 
                                save_path=os.path.join(output_dir, 'diffusion_animation.gif'), 
                                fps=1)
    else:
        print("\nSkipping GIF generation (--no-gif flag set)")
    
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"Total agents: {sim.num_agents}")
    print(f"Influencers: {sim.num_influencers}")
    print(f"Regular agents: {sim.num_regular}")
    print(f"Bots: {sim.num_bots}")
    print(f"\nAverage reach: {np.mean(sim.history['reach']):.2%}")
    print(f"Average forwarding rate: {np.mean(sim.history['forwarding_rate']):.2%}")
    print(f"Average misinformation contamination: {np.mean(sim.history['misinformation_contamination']):.2%}")
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  - network_diffusion_political_bias.png")
    print(f"  - network_diffusion_initial_reputation.png")
    print(f"  - network_diffusion_agent_types.png")
    print(f"  - network_diffusion_message_diffusion.png")
    print(f"  - metrics.png")
    if not args.no_gif:
        print(f"  - diffusion_animation.gif")
    print("="*60)


if __name__ == '__main__':
    main()

