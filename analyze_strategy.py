import os
import json
import numpy as np
import matplotlib.pyplot as plt
from royal_poker.utils import load_strategy
from config import Config

def analyze_strategy_problem():
    """Analyze the betting behavior problem in the current strategies"""
    print("Analyzing strategy behavior problem...")
    
    # Load all strategies for comparison
    strategy_files = [f for f in os.listdir('strategy') if f.endswith('_strategy.json')]
    strategies = {}
    
    for file in strategy_files:
        name = file.replace('_strategy.json', '')
        strategies[name] = load_strategy(os.path.join('strategy', file))
    
    # Focus on problematic states
    print("\n=== JACK RESPONSE TO BET ===")
    for name, strategy in strategies.items():
        # Look for states where Jack (bucket 0) is responding to a bet
        for key, probs in strategy.items():
            bucket, history = key
            if bucket == 0 and len(history) > 0 and history[-1] == 'bet':
                fold_prob = probs[0]  # "check" means fold when responding to bet
                call_prob = probs[1]  # "bet" means call when responding to bet
                print(f"{name}: Jack fold={fold_prob:.6f}, call={call_prob:.6f}")
                
                # This is the problematic state - Jack should usually fold
                if call_prob > 0.2:
                    print(f"  WARNING: Strategy {name} has Jack calling too frequently!")
    
    # Plot the problem for visualization
    plt.figure(figsize=(12, 6))
    
    names = []
    call_probs = []
    
    for name, strategy in strategies.items():
        for key, probs in strategy.items():
            bucket, history = key
            if bucket == 0 and len(history) > 0 and history[-1] == 'bet':
                names.append(name)
                call_probs.append(probs[1])  # "bet" means call
    
    plt.bar(names, call_probs)
    plt.title('Probability of Jack Calling a Bet (Should be Low)')
    plt.ylabel('Probability')
    plt.axhline(y=0.1, color='r', linestyle='--', label='Reasonable threshold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs('analysis', exist_ok=True)
    plt.savefig('analysis/jack_calling_problem.png')
    print("Analysis saved to analysis/jack_calling_problem.png")
    
    # Suggest a fix
    print("\nSUGGESTED FIX:")
    print("The strategy files appear to have incorrect values for Jack responding to bets.")
    print("Consider manually editing the strategy files to fix this issue:")
    print("1. Open each strategy JSON file")
    print('2. Find the key that looks like "(0, (\'bet\',))"')
    print("3. Change the values from [~0.000007, ~0.999993] to [0.95, 0.05]")
    print("This will make the AI fold 95% of the time when it has a Jack and faces a bet.")

if __name__ == "__main__":
    analyze_strategy_problem()
