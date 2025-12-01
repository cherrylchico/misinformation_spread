"""
Analyze and visualize batch simulation results
Loads CSV results, calculates statistics, and generates comparative plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy import stats


def load_results(filepath):
    """Load results from CSV file, skipping comment lines"""
    # Read metadata from comments
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value = line[1:].split(':', 1)
                    metadata[key.strip()] = value.strip()
            else:
                break
    
    # Load data
    df = pd.read_csv(filepath, comment='#')
    
    return df, metadata


def calculate_statistics(df, group_by='param_value'):
    """Calculate mean, std, and confidence intervals"""
    metrics = ['avg_reach', 'avg_forwarding_rate', 'avg_misinformation', 'avg_reputation']
    
    stats_dict = {}
    
    if group_by in df.columns and df[group_by].iloc[0] != 'N/A':
        grouped = df.groupby(group_by)
        
        for metric in metrics:
            stats_dict[metric] = {
                'mean': grouped[metric].mean(),
                'std': grouped[metric].std(),
                'sem': grouped[metric].sem(),  # Standard error of mean
                'count': grouped[metric].count()
            }
            
            # Calculate 95% confidence interval
            ci_95 = grouped[metric].apply(
                lambda x: stats.t.interval(0.95, len(x)-1, loc=np.mean(x), scale=stats.sem(x))
            )
            stats_dict[metric]['ci_lower'] = ci_95.apply(lambda x: x[0])
            stats_dict[metric]['ci_upper'] = ci_95.apply(lambda x: x[1])
    else:
        # No grouping, calculate overall statistics
        for metric in metrics:
            n = len(df[metric])
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            sem_val = stats.sem(df[metric])
            ci = stats.t.interval(0.95, n-1, loc=mean_val, scale=sem_val)
            
            stats_dict[metric] = {
                'mean': mean_val,
                'std': std_val,
                'sem': sem_val,
                'ci_lower': ci[0],
                'ci_upper': ci[1],
                'count': n
            }
    
    return stats_dict


def plot_comparison(df, metadata, output_path=None):
    """Create comprehensive comparison plots"""
    param_name = metadata.get('Parameter varied', 'none')
    
    # Check if we have parameter variation
    if param_name == 'None' or df['param_value'].iloc[0] == 'N/A':
        plot_single_config(df, metadata, output_path)
        return
    
    # Calculate statistics
    stats_dict = calculate_statistics(df)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Effect of {param_name} on Misinformation Spread', 
                 fontsize=14, fontweight='bold')
    
    metrics = [
        ('avg_reach', 'Average Reach', axes[0, 0]),
        ('avg_forwarding_rate', 'Average Forwarding Rate', axes[0, 1]),
        ('avg_misinformation', 'Average Misinformation', axes[1, 0]),
        ('avg_reputation', 'Average Reputation', axes[1, 1])
    ]
    
    param_values = sorted(df['param_value'].unique())
    
    for metric_key, metric_label, ax in metrics:
        stat = stats_dict[metric_key]
        
        # Plot mean with error bars (95% CI)
        ax.errorbar(param_values, stat['mean'], 
                   yerr=[stat['mean'] - stat['ci_lower'], 
                         stat['ci_upper'] - stat['mean']],
                   marker='o', linewidth=2, markersize=8, capsize=5,
                   label='Mean ± 95% CI')
        
        # Add individual points (semi-transparent)
        for pv in param_values:
            subset = df[df['param_value'] == pv][metric_key]
            ax.scatter([pv] * len(subset), subset, alpha=0.2, s=20, color='gray')
        
        ax.set_xlabel(param_name, fontsize=11)
        ax.set_ylabel(metric_label, fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_single_config(df, metadata, output_path=None):
    """Plot distribution of results for a single configuration"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Distribution of Simulation Results', fontsize=14, fontweight='bold')
    
    metrics = [
        ('avg_reach', 'Average Reach', axes[0, 0]),
        ('avg_forwarding_rate', 'Average Forwarding Rate', axes[0, 1]),
        ('avg_misinformation', 'Average Misinformation', axes[1, 0]),
        ('avg_reputation', 'Average Reputation', axes[1, 1])
    ]
    
    for metric_key, metric_label, ax in metrics:
        data = df[metric_key]
        
        # Histogram
        ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
        
        # Add mean line
        mean_val = data.mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.4f}')
        
        # Add std lines
        std_val = data.std()
        ax.axvline(mean_val - std_val, color='orange', linestyle=':', linewidth=1.5,
                  label=f'±1 SD: {std_val:.4f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle=':', linewidth=1.5)
        
        ax.set_xlabel(metric_label, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(metric_label, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()
    
    plt.close()


def print_summary_table(df, metadata):
    """Print formatted summary statistics table"""
    param_name = metadata.get('Parameter varied', 'none')
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Parameter varied: {param_name}")
    print(f"Total simulations: {len(df)}")
    print(f"{'='*80}\n")
    
    if param_name != 'None' and df['param_value'].iloc[0] != 'N/A':
        stats_dict = calculate_statistics(df)
        
        # Create summary table
        param_values = sorted(df['param_value'].unique())
        
        print(f"{'Parameter':<12} {'Reach':<20} {'Forwarding':<20} {'Misinformation':<20}")
        print(f"{'-'*80}")
        
        for pv in param_values:
            reach_mean = stats_dict['avg_reach']['mean'][pv]
            reach_ci = (stats_dict['avg_reach']['ci_lower'][pv], 
                       stats_dict['avg_reach']['ci_upper'][pv])
            
            fwd_mean = stats_dict['avg_forwarding_rate']['mean'][pv]
            fwd_ci = (stats_dict['avg_forwarding_rate']['ci_lower'][pv],
                     stats_dict['avg_forwarding_rate']['ci_upper'][pv])
            
            mis_mean = stats_dict['avg_misinformation']['mean'][pv]
            mis_ci = (stats_dict['avg_misinformation']['ci_lower'][pv],
                     stats_dict['avg_misinformation']['ci_upper'][pv])
            
            print(f"{pv:<12.2f} {reach_mean:.4f} [{reach_ci[0]:.4f}, {reach_ci[1]:.4f}] "
                  f"{fwd_mean:.4f} [{fwd_ci[0]:.4f}, {fwd_ci[1]:.4f}] "
                  f"{mis_mean:.4f} [{mis_ci[0]:.4f}, {mis_ci[1]:.4f}]")
        
        print(f"{'='*80}\n")
    else:
        # Single configuration
        metrics = ['avg_reach', 'avg_forwarding_rate', 'avg_misinformation', 'avg_reputation']
        labels = ['Reach', 'Forwarding Rate', 'Misinformation', 'Reputation']
        
        print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'95% CI':<25}")
        print(f"{'-'*80}")
        
        for metric, label in zip(metrics, labels):
            mean_val = df[metric].mean()
            std_val = df[metric].std()
            n = len(df[metric])
            ci = stats.t.interval(0.95, n-1, loc=mean_val, scale=stats.sem(df[metric]))
            
            print(f"{label:<20} {mean_val:<12.4f} {std_val:<12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
        
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze batch simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze and plot results
  python analyze_results.py \\
    --input results/prob_truth_experiment.csv \\
    --output figures/prob_truth_analysis.png
  
  # Just print statistics
  python analyze_results.py \\
    --input results/experiment.csv \\
    --no-plot
"""
    )
    
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file with batch results')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for plot (default: show plot)')
    parser.add_argument('--no-plot', action='store_true',
                       help='Skip plotting, only print statistics')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.input}")
    df, metadata = load_results(args.input)
    
    print(f"Loaded {len(df)} simulation results")
    print(f"Metadata: {metadata}")
    
    # Print summary statistics
    print_summary_table(df, metadata)
    
    # Generate plots
    if not args.no_plot:
        plot_comparison(df, metadata, args.output)


if __name__ == '__main__':
    main()
