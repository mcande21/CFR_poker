import json
import numpy as np
import os
from collections import defaultdict

def save_strategy(strategy, filename):
    """Save a strategy to a file"""
    strategy_dict = {}
    for k, v in strategy.items():
        if isinstance(k, tuple):
            strategy_dict[str(k)] = v.tolist()
    
    with open(filename, 'w') as f:
        json.dump(strategy_dict, f)

def load_strategy(filename):
    """Load a strategy from a file"""
    # Check if file exists in strategy directory
    if not os.path.exists(filename) and not filename.startswith('strategy/'):
        alternative_path = os.path.join('strategy', os.path.basename(filename))
        if os.path.exists(alternative_path):
            filename = alternative_path
    
    with open(filename, 'r') as f:
        strategy_dict = json.load(f)
    
    # Convert back to defaultdict with tuple keys
    strategy = defaultdict(lambda: np.ones(2)/2)
    for k, v in strategy_dict.items():
        # Parse string representation of tuple back to tuple
        if k.startswith('(') and k.endswith(')'):
            parts = k.strip('()').split(', ', 1)
            if len(parts) == 2:
                card = int(parts[0])
                history_str = parts[1]
                if history_str.startswith('(') and history_str.endswith(')'):
                    history = tuple(item.strip('"\'') for item in history_str.strip('()').split(', '))
                    strategy[(card, history)] = np.array(v)
    
    return strategy