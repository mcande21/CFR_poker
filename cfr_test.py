# Simple test for CFR agent
from royal_poker.env import KuhnPokerEnv
from royal_poker.cfrAgent import CFRAgent
import numpy as np

def run_basic_test():
    """Run a simple test of the CFR agent's basic functionality"""
    print("\n=== BASIC CFR AGENT TEST ===\n")
    
    # Create environment and agent
    env = KuhnPokerEnv()
    agent = CFRAgent(env=env, epsilon=0.1, decay=0.99)
    
    # Test making decisions
    print("Testing basic decisions...")
    
    # Test with Jack (0)
    state = (0, ())  # Jack, initial decision
    action = agent.choose(state, ["check", "bet"])
    print(f"Jack, initial decision: chose {action}")
    
    # Test with King (2)
    state = (2, ())  # King, initial decision
    action = agent.choose(state, ["check", "bet"])
    print(f"King, initial decision: chose {action}")
    
    # Test facing a bet with Jack
    state = (0, ("bet",))  # Jack, facing a bet
    action = agent.choose(state, ["call", "fold"])
    print(f"Jack, facing a bet: chose {action}")
    
    # Test a simple update
    print("\nTesting simple update...")
    state = (1, ())  # Queen, initial decision
    action = "check"
    reward = -0.5  # Lost after checking
    
    # Show strategy before update
    strategy_before = agent.get_strategy(state)
    print(f"Strategy before update: Check={strategy_before[0]:.4f}, Bet={strategy_before[1]:.4f}")
    
    # Update
    agent.update(state, action, reward)
    
    # Show strategy after update
    strategy_after = agent.get_strategy(state)
    print(f"Strategy after update: Check={strategy_after[0]:.4f}, Bet={strategy_after[1]:.4f}")
    
    # Test playing a full game
    print("\nPlaying a full game...")
    env.reset()
    
    # Set specific cards for demonstration
    env.hands = [1, 2]  # Player has Queen (1), opponent has King (2)
    print(f"Player card: Queen (1), Opponent card: King (2)")
    
    done = False
    total_reward = 0
    
    # Player's turn first
    state = env.state()
    action = agent.choose(state, env.get_legal_actions())
    print(f"Player chose: {action}")
    next_state, _, done = env.step(action)
    
    # Opponent's turn if not done
    if not done:
        state = env.state()
        action = agent.choose(state, env.get_legal_actions())
        print(f"Opponent chose: {action}")
        next_state, _, done = env.step(action)
    
    # Game might continue
    while not done:
        state = env.state()
        action = agent.choose(state, env.get_legal_actions())
        print(f"Next action: {action}")
        next_state, _, done = env.step(action)
    
    # Game over
    reward = env.get_payoff(0)  # Get payoff from player 0's perspective
    print(f"Game over. Final reward: {reward}")
    
    # Check if we can save and load a strategy
    print("\nTesting strategy save/load...")
    test_file = "test_strategy.json"
    
    # Save the strategy
    agent.save_strategy(test_file)
    
    # Create a new agent and load the strategy
    new_agent = CFRAgent(env=env)
    new_agent.load_strategy(test_file)
    
    # Compare a decision to make sure it loaded properly
    state = (2, ())  # King, initial
    action1 = agent.choose(state, ["check", "bet"])
    action2 = new_agent.choose(state, ["check", "bet"])
    
    print(f"Original agent chooses: {action1}")
    print(f"Agent with loaded strategy chooses: {action2}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    # Set seed for reproducibility
    np.random.seed(42)
    run_basic_test()