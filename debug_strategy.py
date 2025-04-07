import json
import os
import sys
from config import Config

def print_strategy_breakdown(strategy_path):
    """Print a human-readable breakdown of a strategy file"""
    print(f"Analyzing strategy from: {strategy_path}")
    
    # Load the strategy
    with open(strategy_path, 'r') as f:
        strategy = json.load(f)
    
    print(f"Loaded strategy with {len(strategy)} states")
    
    # Prepare bucket descriptions
    bucket_ranges = {}
    if Config.BUCKETS <= Config.CARD_VALUES:
        # Each bucket is a card type (J, Q, K)
        card_names = ["Jack", "Queen", "King"]
        for bucket in range(min(Config.BUCKETS, Config.CARD_VALUES)):
            bucket_ranges[bucket] = f"{card_names[bucket]}"
    else:
        # Buckets represent positions in the deck
        # For a 100-card deck with 33 of each value, describe the buckets
        bucket_size = Config.DECK_SIZE // Config.BUCKETS
        for bucket in range(Config.BUCKETS):
            start = bucket * bucket_size
            end = (bucket + 1) * bucket_size - 1 if bucket < Config.BUCKETS - 1 else Config.DECK_SIZE - 1
            bucket_ranges[bucket] = f"Cards #{start}-{end} (J,Q,K mix)"
    
    # Group by bucket
    for bucket, description in bucket_ranges.items():
        print(f"\n=== STRATEGY FOR {description} ===")
        
        # Find all states for this bucket
        relevant_states = {}
        for state, probs in strategy.items():
            if state.startswith(f"({bucket},"):
                # Extract history
                history_part = state.split(", ", 1)[1].strip(")")
                relevant_states[history_part] = probs
        
        # Sort by history length for readability
        for history, probs in sorted(relevant_states.items(), key=lambda x: len(x[0])):
            # Format the probabilities
            check_prob = probs[0]
            bet_prob = probs[1]
            
            # Show in a readable format
            print(f"History: {history:20} → Check: {check_prob:.4f}, Bet: {bet_prob:.4f}")
    
    # Specific analysis of post-check strategies
    print("\n=== POST-CHECK STRATEGIES BY BUCKET ===")
    for bucket in range(min(Config.BUCKETS, 10)):  # Only show first 10 buckets
        bucket_desc = bucket_ranges.get(bucket, f"Bucket {bucket}")
        key = f"({bucket}, ('check',))"
        if key in strategy:
            probs = strategy[key]
            print(f"{bucket_desc}: After opponent checks → Check: {probs[0]:.4f}, Bet: {probs[1]:.4f}")
        else:
            print(f"{bucket_desc}: No explicit post-check strategy found")

def main():
    if len(sys.argv) > 1:
        # Use provided strategy file
        strategy_path = sys.argv[1]
    else:
        # Default to 10B_CFR+ strategy (better for 100-card deck)
        strategy_path = os.path.join("strategy", "10B_CFR+_strategy.json")
    
    # Ensure the path exists
    if not os.path.exists(strategy_path):
        # Try prepending the strategy folder
        if not strategy_path.startswith("strategy/"):
            new_path = os.path.join("strategy", os.path.basename(strategy_path))
            if os.path.exists(new_path):
                strategy_path = new_path
            else:
                # List available strategies
                print(f"Strategy {strategy_path} not found.")
                print("Available strategies:")
                for f in os.listdir("strategy"):
                    if f.endswith(".json"):
                        print(f"  - strategy/{f}")
                return
    
    print_strategy_breakdown(strategy_path)

if __name__ == "__main__":
    main()
