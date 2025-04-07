import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

def load_results(filepath='results/experiment_results.json'):
    """Load experiment results from the JSON file"""
    with open(filepath, 'r') as f:
        results = json.load(f)
    return results

def visualize_exploitability(results):
    """Visualize exploitability comparison across different strategies"""
    # Extract exploitability data
    strategies = []
    exploitability = []
    
    for strategy_name, data in results.items():
        if strategy_name != "comparisons":
            strategies.append(strategy_name)
            exploitability.append(data["exploitability"])
    
    # Sort by bucket size then variant
    strategies_sorted = sorted(strategies, key=lambda x: (int(x.split('B_')[0]), x.split('B_')[1]))
    exploitability_sorted = [results[s]["exploitability"] for s in strategies_sorted]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    bars = plt.bar(strategies_sorted, exploitability_sorted, color=sns.color_palette("viridis", len(strategies_sorted)))
    
    # Add labels and title
    plt.title('Strategy Exploitability Comparison', fontsize=16)
    plt.xlabel('Strategy', fontsize=14)
    plt.ylabel('Exploitability', fontsize=14)
    plt.xticks(rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('visualization/exploitability_comparison.png')
    plt.close()
    
    print(f"Exploitability comparison chart saved to 'results/exploitability_comparison.png'")

def visualize_head_to_head(results):
    """Create a heatmap of head-to-head performance"""
    comparisons = results["comparisons"]
    
    # Convert to DataFrame for easier visualization
    strategies = sorted(comparisons.keys(), key=lambda x: (int(x.split('B_')[0]), x.split('B_')[1]))
    data = []
    
    for s1 in strategies:
        row = []
        for s2 in strategies:
            if s1 == s2:
                row.append(0)  # No self-comparison
            else:
                row.append(comparisons[s1][s2])
        data.append(row)
    
    df = pd.DataFrame(data, index=strategies, columns=strategies)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create heatmap
    sns.heatmap(df, annot=True, cmap=cmap, center=0, 
                linewidths=.5, fmt=".3f", cbar_kws={'label': 'Expected Payoff'})
    
    plt.title('Head-to-Head Strategy Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualization/head_to_head_comparison.png')
    plt.close()
    
    print(f"Head-to-head comparison heatmap saved to 'results/head_to_head_comparison.png'")

def visualize_bucket_variant_comparison(results):
    """Compare performance by bucket size and variant"""
    # Extract data
    buckets = [3, 10, 25]
    variants = ["Vanilla", "CFR+"]
    
    # Calculate average performance for each bucket/variant
    performance_data = {}
    
    for bucket in buckets:
        performance_data[bucket] = {}
        for variant in variants:
            strategy_name = f"{bucket}B_{variant}"
            # Calculate average performance against all other strategies
            avg_performance = np.mean([v for k, v in results["comparisons"][strategy_name].items()])
            performance_data[bucket][variant] = avg_performance
    
    # Convert to DataFrame
    df = pd.DataFrame(performance_data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Bar chart
    df.plot(kind='bar', ax=plt.gca())
    
    plt.title('Average Strategy Performance by Bucket Size', fontsize=16)
    plt.xlabel('CFR Variant', fontsize=14)
    plt.ylabel('Average Expected Payoff', fontsize=14)
    plt.xticks(rotation=0)
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.legend(title='Bucket Size')
    plt.tight_layout()
    plt.savefig('visualization/bucket_variant_comparison.png')
    plt.close()
    
    print(f"Bucket and variant comparison chart saved to 'results/bucket_variant_comparison.png'")

def visualize_strategy_distribution(directory='strategy'):
    """Visualize the action distributions in the strategies"""
    strategy_files = [f for f in os.listdir(directory) if f.endswith('_strategy.json')]
    
    # For simplicity, let's analyze check/bet probabilities for the initial state
    initial_states = {}
    
    for filename in strategy_files:
        strategy_name = filename.replace('_strategy.json', '')
        
        with open(os.path.join(directory, filename), 'r') as f:
            strategy = json.load(f)
        
        # Look for initial states (with empty history)
        for key, value in strategy.items():
            if ", ()" in key:  # Initial state with empty history
                bucket = int(key.split(',')[0].strip('('))
                initial_states.setdefault(strategy_name, {})[bucket] = value
    
    # Create a visualization for each strategy
    for strategy_name, states in initial_states.items():
        buckets = sorted(states.keys())
        check_probs = [states[b][0] for b in buckets]
        bet_probs = [states[b][1] for b in buckets]
        
        plt.figure(figsize=(10, 6))
        
        width = 0.35
        x = np.arange(len(buckets))
        
        plt.bar(x - width/2, check_probs, width, label='Check Probability')
        plt.bar(x + width/2, bet_probs, width, label='Bet Probability')
        
        plt.xlabel('Card Bucket', fontsize=14)
        plt.ylabel('Probability', fontsize=14)
        plt.title(f'Action Probabilities for Initial State - {strategy_name}', fontsize=16)
        plt.xticks(x, buckets)
        plt.ylim(0, 1)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'visualization/{strategy_name}_initial_actions.png')
        plt.close()
        
        print(f"Strategy distribution chart saved to 'results/{strategy_name}_initial_actions.png'")

def main():
    # Make sure the results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Load results
    results = load_results()
    
    # Generate visualizations
    visualize_exploitability(results)
    visualize_head_to_head(results)
    visualize_bucket_variant_comparison(results)
    visualize_strategy_distribution()
    
    print("All visualizations completed!")

if __name__ == "__main__":
    main()
