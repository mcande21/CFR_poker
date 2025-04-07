import json
import numpy as np
import os

def hand_strength(card: int) -> int:
    """Direct card value comparison for Kuhn"""
    return card

def action_translation(abstract_action: str, is_response_to_bet: bool = False) -> str:
    """Translate abstract actions to concrete game actions based on context"""
    if not is_response_to_bet:
        return abstract_action  # No translation needed
    else:
        # When responding to a bet, "check" means "fold" and "bet" means "call"
        if abstract_action == "check":
            return "fold"
        else:  # abstract_action == "bet"
            return "call"

def save_strategy(strategy, filename):
    """Save a strategy to a file"""
    # Convert defaultdict to regular dict for JSON serialization
    strategy_dict = {}
    for k, v in strategy.items():
        if isinstance(k, tuple):
            # Convert tuple keys to strings
            str_key = str(k)
            strategy_dict[str_key] = v.tolist()
    
    # Add some debugging info
    print(f"Saving strategy with {len(strategy_dict)} states to {filename}")
    sample_keys = list(strategy_dict.keys())[:3]
    print(f"Sample strategy keys: {sample_keys}")
    
    with open(filename, 'w') as f:
        json.dump(strategy_dict, f)

def load_strategy(filename):
    """Load a strategy from a file"""
    from collections import defaultdict
    
    # Handle both old and new file paths
    if not os.path.exists(filename) and not filename.startswith('strategy/'):
        # Try looking in the strategy directory
        alternative_path = os.path.join('strategy', os.path.basename(filename))
        if os.path.exists(alternative_path):
            filename = alternative_path
    
    print(f"Loading strategy from: {filename}")
    with open(filename, 'r') as f:
        strategy_dict = json.load(f)
    
    print(f"Loaded strategy with {len(strategy_dict)} states")
    
    # Convert back to defaultdict with tuple keys
    strategy = defaultdict(lambda: np.ones(2)/2)
    for k, v in strategy_dict.items():
        # Parse string representation of tuple back to tuple
        # Format: "(bucket, (history))"
        if k.startswith('(') and k.endswith(')'):
            parts = k.strip('()').split(', ', 1)
            if len(parts) == 2:
                bucket = int(parts[0])
                # Parse the history tuple
                history_str = parts[1]
                if history_str.startswith('(') and history_str.endswith(')'):
                    history = tuple(item.strip('"\'') for item in history_str.strip('()').split(', '))
                    strategy[(bucket, history)] = np.array(v)
    
    # Print some sample strategies for debugging
    sample_keys = list(strategy.keys())[:3]
    for key in sample_keys:
        print(f"Strategy for {key}: {strategy[key]}")
    
    # Log contextual actions to understand strategy behavior
    print("\nContextual action probabilities:")
    for key in strategy.keys():
        bucket, history = key
        if bucket == 0:  # Focus on Jack (lowest card)
            context = ""
            if len(history) > 0:
                if history[-1] == "bet":
                    context = " (responding to bet: [fold, call])"
                elif history[-1] == "check":
                    context = " (responding to check: [check, bet])"
            if not history:
                context = " (first action: [check, bet])"
            print(f"Jack with history {history}{context}: {strategy[key]}")
    
    return strategy