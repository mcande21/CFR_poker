# Test script for RuleBased agent
import numpy as np
from random import seed
from environment import KuhnPokerEnv
from Agents import RuleBasedAgent

def test_rule_based_agent():
    """Test the RuleBased agent"""
    print("Testing RuleBased agent...")
    
    # Create environment
    env = KuhnPokerEnv()
    
    # Initialize agent
    agent = RuleBasedAgent(name="TestRuleAgent")
    
    # Test agent decision making for some sample states
    test_states = [
        # Initial state with Jack, Queen, and King
        ((0, ()), ["check", "bet"]),  # Jack with no history
        ((1, ()), ["check", "bet"]),  # Queen with no history
        ((2, ()), ["check", "bet"]),  # King with no history
        
        # Responding to a check
        ((0, ("check",)), ["check", "bet"]),  # Jack after opponent check
        ((1, ("check",)), ["check", "bet"]),  # Queen after opponent check
        ((2, ("check",)), ["check", "bet"]),  # King after opponent check
        
        # Responding to a bet
        ((0, ("bet",)), ["call", "fold"]),  # Jack facing a bet
        ((1, ("bet",)), ["call", "fold"]),  # Queen facing a bet
        ((2, ("bet",)), ["call", "fold"]),  # King facing a bet
    ]
    
    print("\nTesting agent decisions for different game states:")
    
    # Test multiple times to see probabilities in action
    print("\nFirst run (showing decision patterns):")
    for state, legal_actions in test_states:
        action = agent.choose(state, legal_actions)
        card, history = state
        card_name = "Jack" if card == 0 else "Queen" if card == 1 else "King"
        print(f"{card_name} with history {history}: chose {action}")
    
    # Test multiple sampling to see action distribution
    print("\nSampling action distributions:")
    
    # Create a dictionary to track action counts for interesting states
    action_counts = {
        "King_initial": {"check": 0, "bet": 0},
        "Queen_initial": {"check": 0, "bet": 0},
        "Jack_initial": {"check": 0, "bet": 0},
        "King_vs_bet": {"call": 0, "fold": 0},
        "Jack_vs_bet": {"call": 0, "fold": 0}
    }
    
    # Sample multiple times to see distribution
    n_samples = 1000
    for _ in range(n_samples):
        # King initial action
        action = agent.choose((2, ()), ["check", "bet"])
        action_counts["King_initial"][action] += 1
        
        # Queen initial action
        action = agent.choose((1, ()), ["check", "bet"])
        action_counts["Queen_initial"][action] += 1
        
        # Jack initial action
        action = agent.choose((0, ()), ["check", "bet"])
        action_counts["Jack_initial"][action] += 1
        
        # King vs bet
        action = agent.choose((2, ("bet",)), ["call", "fold"])
        action_counts["King_vs_bet"][action] += 1
        
        # Jack vs bet
        action = agent.choose((0, ("bet",)), ["call", "fold"])
        action_counts["Jack_vs_bet"][action] += 1
    
    # Print distributions
    print(f"\nAction distributions over {n_samples} samples:")
    for state, counts in action_counts.items():
        total = sum(counts.values())
        dist = {a: f"{(c/total)*100:.1f}%" for a, c in counts.items()}
        print(f"{state}: {dist}")
    
    # Test update function (should do nothing for rule-based agent)
    print("\nTesting that update function has no effect:")
    
    # Initial state with Jack
    state = (0, ())
    legal_actions = ["check", "bet"]
    
    # Sample actions before update
    print("Sampling 5 actions before update:")
    for _ in range(5):
        action = agent.choose(state, legal_actions)
        print(f"Action: {action}")
    
    # Simulate action and reward
    action = "check"
    reward = -0.5
    agent.update(state, action, reward)
    
    # Sample actions after update (should be similar distribution)
    print("\nSampling 5 actions after update (should have similar distribution):")
    for _ in range(5):
        action = agent.choose(state, legal_actions)
        print(f"Action: {action}")

    print("\nTesting complete!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    seed(42)
    np.random.seed(42)
    
    # Run tests
    test_rule_based_agent()