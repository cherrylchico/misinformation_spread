"""
Script to compare results from multiple batch simulation experiments.
Overlays curves (mean + 95% CI) from different CSV files on the same plot.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np
from scipy import stats

def load_and_process_data(filepath, label):
    """Load CSV and calculate stats per parameter value"""
    try:
        # Read CSV, skipping comment lines
        df = pd.read_csv(filepath, comment='#')
        df['Experiment'] = label
        return df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def plot_comparison(dfs, metric, output_file, title=None):
    """Plot comparison of multiple experiments for a given metric"""
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Set style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot using seaborn lineplot which automatically calculates mean and CI
    # ci=95 is the default, but being explicit
    ax = sns.lineplot(
        data=combined_df,
        x="param_value",
        y=metric,
        hue="Experiment",
        style="Experiment",
        markers=True,
        dashes=False,
        errorbar=('ci', 95),
        err_style="band",  # "band" for shaded area, "bars" for error bars
        palette="viridis",
        linewidth=2.5,
        marker='o',
        markersize=8
    )
    
    # Customize labels
    metric_names = {
        'avg_reach': 'Average Reach',
        'avg_forwarding_rate': 'Average Forwarding Rate',
        'avg_misinformation': 'Misinformation Contamination',
        'avg_reputation': 'Average Reputation'
    }
    
    y_label = metric_names.get(metric, metric.replace('_', ' ').title())
    
    plt.xlabel("Parameter Value (e.g., Count, Probability)", fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    
    if title:
        plt.title(title, fontsize=14, pad=15)
    else:
        plt.title(f"Comparison of {y_label}", fontsize=14, pad=15)
        
    plt.legend(title="Experiment", fontsize=10, title_fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple batch simulation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python code/compare_experiments.py \\
    --inputs results/exp_bots.csv results/exp_influencers.csv \\
    --labels "Varying Bots" "Varying Influencers" \\
    --metric avg_reach \\
    --output figures/comparison_reach.png
"""
    )
    
    parser.add_argument('--inputs', type=str, nargs='+', required=True,
                       help='List of input CSV files')
    parser.add_argument('--labels', type=str, nargs='+', required=True,
                       help='List of labels for the legend (must match number of inputs)')
    parser.add_argument('--metric', type=str, default='avg_reach',
                       choices=['avg_reach', 'avg_forwarding_rate', 'avg_misinformation', 'avg_reputation'],
                       help='Metric to plot on Y-axis')
    parser.add_argument('--output', type=str, required=True,
                       help='Output image file path')
    parser.add_argument('--title', type=str, default=None,
                       help='Custom title for the plot')
    
    args = parser.parse_args()
    
    if len(args.inputs) != len(args.labels):
        print("Error: Number of input files must match number of labels.")
        return
    
    # Load data
    dfs = []
    for filepath, label in zip(args.inputs, args.labels):
        if os.path.exists(filepath):
            print(f"Loading: {filepath} as '{label}'")
            df = load_and_process_data(filepath, label)
            if df is not None:
                dfs.append(df)
        else:
            print(f"Warning: File not found: {filepath}")
    
    if not dfs:
        print("No data loaded. Exiting.")
        return
        
    # Plot
    plot_comparison(dfs, args.metric, args.output, args.title)

if __name__ == '__main__':
    main()
