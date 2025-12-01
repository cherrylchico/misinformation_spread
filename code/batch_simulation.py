"""
Batch simulation runner for misinformation spread experiments
Runs multiple simulations with parameter variations and saves results
"""

import numpy as np
import pandas as pd
import json
import argparse
import os
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(__file__))

from snlearn.simulation import Simulation


def run_single_simulation(config_sim, config_agent, config_influencer=None, 
                         config_bot=None, seed=None):
    """Run a single simulation and return summary statistics"""
    if seed is not None:
        np.random.seed(seed)
    
    sim = Simulation(config_sim, config_agent, config_influencer, config_bot)
    results = sim.run()
    
    # Calculate summary statistics
    stats = {
        'avg_reach': np.mean(sim.history['reach']),
        'std_reach': np.std(sim.history['reach']),
        'avg_forwarding_rate': np.mean(sim.history['forwarding_rate']),
        'std_forwarding_rate': np.std(sim.history['forwarding_rate']),
        'avg_misinformation': np.mean(sim.history['misinformation_contamination']),
        'std_misinformation': np.std(sim.history['misinformation_contamination']),
        'avg_reputation': np.mean(sim.history['average_reputation']),
        'std_reputation': np.std(sim.history['average_reputation']),
        'final_reputation': sim.history['average_reputation'][-1],
        'num_agents': sim.num_agents,
        'num_influencers': sim.num_influencers,
        'num_bots': sim.num_bots,
        'num_regular': sim.num_regular
    }
    
    return stats


def create_config_variation(base_config, param_name, param_value):
    """Create a modified config with a specific parameter value"""
    config = base_config.copy()
    
    # Handle nested parameters (e.g., message.prob_truth)
    if '.' in param_name:
        parts = param_name.split('.')
        if parts[0] not in config:
            config[parts[0]] = {}
        config[parts[0]][parts[1]] = param_value
    else:
        config[param_name] = param_value
    
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Run batch simulations with parameter variations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Vary prob_truth with 100 runs per value
  python batch_simulation.py --num-runs 100 \\
    --vary message.prob_truth --values 0.1 0.3 0.5 0.7 0.9 \\
    --config-sim config/config_sim_general.json \\
    --config-agent config/config_sim_agent.json \\
    --output results/prob_truth_experiment.csv
  
  # Compare with/without bots
  python batch_simulation.py --num-runs 50 \\
    --vary num_bots --values 0 5 10 20 \\
    --config-sim config/config_sim_general.json \\
    --config-agent config/config_sim_agent.json \\
    --config-bot config/config_sim_bot.json \\
    --output results/bots_experiment.csv
  
  # Fixed seed for reproducibility
  python batch_simulation.py --num-runs 100 \\
    --vary message.prob_truth --values 0.5 \\
    --config-sim config/config_sim_general.json \\
    --config-agent config/config_sim_agent.json \\
    --seed 42 \\
    --output results/baseline.csv
"""
    )
    
    # Required arguments
    parser.add_argument('--num-runs', type=int, required=True,
                       help='Number of simulation runs per parameter value')
    parser.add_argument('--config-sim', type=str, required=True,
                       help='Path to simulation config JSON file')
    parser.add_argument('--config-agent', type=str, required=True,
                       help='Path to agent config JSON file')
    
    # Optional arguments
    parser.add_argument('--config-influencer', type=str, default=None,
                       help='Path to influencer config JSON file (optional)')
    parser.add_argument('--config-bot', type=str, default=None,
                       help='Path to bot config JSON file (optional)')
    parser.add_argument('--vary', type=str, default=None,
                       help='Parameter to vary (e.g., message.prob_truth, num_bots)')
    parser.add_argument('--values', type=float, nargs='+', default=[None],
                       help='Values to test for the varied parameter')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (CSV format)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Base random seed (each run uses seed+run_id)')
    parser.add_argument('--comment', type=str, default='',
                       help='Comment to add to output file header')
    
    args = parser.parse_args()
    
    # Load base configurations
    print(f"Loading configurations...")
    with open(args.config_sim, 'r') as f:
        base_config_sim = json.load(f)
    
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
    
    # Prepare results storage
    results_list = []
    
    # Total number of simulations
    total_sims = args.num_runs * len(args.values)
    
    print(f"\n{'='*60}")
    print(f"BATCH SIMULATION")
    print(f"{'='*60}")
    print(f"Parameter to vary: {args.vary or 'None (fixed parameters)'}")
    print(f"Values: {args.values}")
    print(f"Runs per value: {args.num_runs}")
    print(f"Total simulations: {total_sims}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Run simulations
    with tqdm(total=total_sims, desc="Running simulations") as pbar:
        for param_value in args.values:
            # Create config for this parameter value
            if args.vary and param_value is not None:
                config_sim = create_config_variation(base_config_sim, args.vary, param_value)
                
                # Special handling for num_bots and num_influencers
                # Simulation class prioritizes count in specific configs, so we must update them too
                if args.vary == 'num_bots' and config_bot:
                    config_bot = config_bot.copy()
                    config_bot['count'] = int(param_value)
                
                if args.vary == 'num_influencers' and config_influencer:
                    config_influencer = config_influencer.copy()
                    config_influencer['count'] = int(param_value)
            else:
                config_sim = base_config_sim
            
            # Run multiple simulations with this configuration
            for run_id in range(args.num_runs):
                # Set seed if provided
                run_seed = None
                if args.seed is not None:
                    run_seed = args.seed + run_id
                
                # Run simulation
                stats = run_single_simulation(
                    config_sim, config_agent, config_influencer, config_bot, run_seed
                )
                
                # Add metadata
                stats['run_id'] = run_id
                stats['param_name'] = args.vary or 'none'
                stats['param_value'] = param_value if param_value is not None else 'N/A'
                stats['seed'] = run_seed if run_seed is not None else 'random'
                
                results_list.append(stats)
                pbar.update(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(results_list)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Save results with metadata header
    with open(args.output, 'w') as f:
        # Write metadata as comments
        f.write(f"# Batch Simulation Results\n")
        f.write(f"# Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Parameter varied: {args.vary or 'None'}\n")
        f.write(f"# Values: {args.values}\n")
        f.write(f"# Runs per value: {args.num_runs}\n")
        f.write(f"# Total simulations: {total_sims}\n")
        if args.comment:
            f.write(f"# Comment: {args.comment}\n")
        f.write(f"#\n")
        
        # Write CSV data
        df.to_csv(f, index=False)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SAVED")
    print(f"{'='*60}")
    print(f"Output file: {args.output}")
    print(f"Total rows: {len(df)}")
    print(f"\nSummary statistics:")
    
    # Group by parameter value and show summary
    if args.vary and args.values[0] is not None:
        summary = df.groupby('param_value').agg({
            'avg_reach': ['mean', 'std'],
            'avg_forwarding_rate': ['mean', 'std'],
            'avg_misinformation': ['mean', 'std']
        })
        print(summary)
    else:
        print(f"Average reach: {df['avg_reach'].mean():.4f} ± {df['avg_reach'].std():.4f}")
        print(f"Average forwarding: {df['avg_forwarding_rate'].mean():.4f} ± {df['avg_forwarding_rate'].std():.4f}")
        print(f"Average misinformation: {df['avg_misinformation'].mean():.4f} ± {df['avg_misinformation'].std():.4f}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
