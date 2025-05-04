# Test script for CFR agent
import numpy as np
from environment import KuhnPokerEnv
from Agents import CFRAgent

def test_cfr_agent():
    """Test the CFR agent"""
    print("Testing CFR agent...")
    
    # Create environment
    env = KuhnPokerEnv()
    
    # Initialize agent with environment
    agent = CFRAgent(env=env, epsilon=0.1, decay=0.99)
    
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
    
    for state, legal_actions in test_states:
        action = agent.choose(state, legal_actions)
        card, history = state
        card_name = "Jack" if card == 0 else "Queen" if card == 1 else "King"
        print(f"{card_name} with history {history}: chose {action}")
    
    # Test learning update functionality with a simple scenario
    print("\nTesting agent learning updates:")
    
    # Initial state with Jack
    state = (0, ())
    legal_actions = ["check", "bet"]
    
    # Before update
    print("Strategy before update:")
    probs_before = agent.get_strategy(state)
    print(f"check: {probs_before[0]:.4f}, bet: {probs_before[1]:.4f}")
    
    # Simulate action and reward
    action = "check"
    reward = -0.5  # Negative reward to incentivize trying "bet" instead
    agent.update(state, action, reward)
    
    # After update
    print("Strategy after update:")
    probs_after = agent.get_strategy(state)
    print(f"check: {probs_after[0]:.4f}, bet: {probs_after[1]:.4f}")
    
    # Show regret values
    print(f"Regret values: {agent.regrets[state]}")

    print("\nTesting complete!")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    test_cfr_agent()