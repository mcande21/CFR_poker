# Test script for incremental CFR agent implementation
import os
import numpy as np
import matplotlib.pyplot as plt
from royal_poker.env import KuhnPokerEnv
from royal_poker.cfrAgent import CFRAgent
from config import Config
from royal_poker.evaluate import Evaluator

def test_cfr_agent():
    """Test the basic functionality of the incremental CFR agent"""
    print("Testing incremental CFR agent...")
    
    # Create environment
    env = KuhnPokerEnv()
    
    # Initialize agent with environment
    agent = CFRAgent(env=env, epsilon=0.1, decay=0.99)
    
    # Load a pre-trained strategy if available
    strategy_path = os.path.join("strategy", "3B_CFR+_strategy.json")
    if os.path.exists(strategy_path):
        print(f"Loading strategy from: {strategy_path}")
        agent.load_strategy(strategy_path)
    
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
        bucket, history = state
        card_name = "Jack" if bucket == 0 else "Queen" if bucket == 1 else "King"
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
    
    # Test saving the strategy
    print("\nTesting strategy saving and loading:")
    test_strategy_path = "test_strategy.json"
    
    # Save current strategy
    agent.save_strategy(test_strategy_path)
    
    # Load it back with a new agent
    new_agent = CFRAgent(env=env)
    new_agent.load_strategy(test_strategy_path)
    
    # Compare strategies
    original_strategy = agent.get_average_strategy()
    loaded_strategy = new_agent.get_average_strategy()
    
    # Check a few states to confirm they match
    for state in [(0, ()), (1, ()), (2, ())]:
        if state in original_strategy and state in loaded_strategy:
            original_probs = original_strategy[state]
            loaded_probs = loaded_strategy[state]
            print(f"State {state}: Original {original_probs}, Loaded {loaded_probs}")
    
    # Clean up test file
    if os.path.exists(test_strategy_path):
        os.remove(test_strategy_path)
        print(f"Removed test file: {test_strategy_path}")
    
def test_self_play_training(iterations):
    """Test the agent training through self-play"""
    print("\nTesting CFR agent self-play training...")
    
    # Create environment and agent
    env = KuhnPokerEnv()
    agent = CFRAgent(env=env, epsilon=0.3, decay=0.999)
    
    # Set up training parameters
    num_episodes = iterations
    
    # Keep track of average rewards
    rewards = []
    total_reward = 0
    
    print(f"Training for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        if episode % 1000 == 0:
            print(f"Episode {episode}/{num_episodes}")
        
        # Reset environment
        env.reset()
        done = False
        episode_reward = 0
        
        # Store states and actions for the update
        states = []
        actions = []
        
        # Play until terminal state
        while not done:
            state = env.state()
            legal_actions = env.get_legal_actions()
            
            # Agent chooses action
            action = agent.choose(state, legal_actions)
            states.append(state)
            actions.append(action)
            
            # Take step in environment
            next_state, _, done = env.step(action)
            
            # If game is over, get reward
            if done:
                reward = env.get_payoff(env.current_player)
                episode_reward = reward
                
                # Update agent for all state-action pairs
                for s, a in zip(states, actions):
                    agent.update(s, a, reward)
        
        # Track rewards
        total_reward += episode_reward
        if episode % 1000 == 0:
            avg_reward = total_reward / (episode + 1)
            rewards.append(avg_reward)
            print(f"Average reward after {episode} episodes: {avg_reward:.4f}")
            print(f"Exploration rate (epsilon): {agent.epsilon:.4f}")
    
    # Save final strategy
    training_strategy_path = os.path.join("strategy", "incremental_cfr_strategy.json")
    agent.save_strategy(training_strategy_path)
    print(f"Saved trained strategy to {training_strategy_path}")
    
    # Print final average reward
    final_avg_reward = total_reward / num_episodes
    print(f"Final average reward: {final_avg_reward:.4f}")
    
    # Print some key strategy probabilities
    print("\nFinal strategy probabilities:")
    for card in range(3):  # Jack, Queen, King
        state = (card, ())
        probs = agent.get_strategy(state)
        card_name = "Jack" if card == 0 else "Queen" if card == 1 else "King"
        print(f"{card_name} initial decision: check={probs[0]:.4f}, bet={probs[1]:.4f}")

def compare_hyperparameters():
    """
    Compare different epsilon and decay rate combinations by training 
    agents with these parameters and evaluating their performance.
    """
    print("\n" + "="*60)
    print("HYPERPARAMETER COMPARISON EXPERIMENT")
    print("="*60)
    
    # Environment setup
    env = KuhnPokerEnv()
    
    # Define hyperparameter combinations to test
    epsilon_values = [0.1, 0.3, 0.5]
    decay_values = [0.9999, 0.999, 0.99]
    
    # Number of training episodes per agent
    num_episodes = 30000
    
    # Create directory for results
    os.makedirs('hyperparameter_test', exist_ok=True)
    
    # Store trained agents and their metrics
    agents = {}
    exploit_histories = {}
    reward_histories = {}
    
    # Train agents with different hyperparameter combinations
    for epsilon in epsilon_values:
        for decay in decay_values:
            # Create agent ID
            agent_id = f"e{epsilon:.1f}_d{decay:.4f}"
            print(f"\nTraining agent with epsilon={epsilon}, decay={decay}")
            
            # Initialize agent
            agent = CFRAgent(env=env, epsilon=epsilon, decay=decay)
            
            # Metrics tracking
            total_reward = 0
            exploit_history = []
            reward_history = []
            best_exploitability = float('inf')
            
            # Training loop
            eval_frequency = min(1000, num_episodes // 10)  # Evaluate ~10 times
            
            for episode in range(num_episodes):
                # Progress indicator
                if episode % 1000 == 0:
                    print(f"  Episode {episode}/{num_episodes}")
                
                # Reset environment
                env.reset()
                done = False
                states = []
                actions = []
                
                # Play until terminal state
                while not done:
                    state = env.state()
                    legal_actions = env.get_legal_actions()
                    action = agent.choose(state, legal_actions)
                    states.append(state)
                    actions.append(action)
                    _, _, done = env.step(action)
                
                # Game over - update agent and track rewards
                reward = env.get_payoff(env.current_player)
                for s, a in zip(states, actions):
                    agent.update(s, a, reward)
                
                total_reward += reward
                
                # Periodic evaluation
                if (episode + 1) % eval_frequency == 0:
                    # Current average reward
                    avg_reward = total_reward / (episode + 1)
                    reward_history.append((episode + 1, avg_reward))
                    
                    # Calculate exploitability
                    current_strategy = agent.get_average_strategy()
                    current_exploit = Evaluator.calculate_exploitability(current_strategy)
                    exploit_history.append((episode + 1, current_exploit))
                    
                    print(f"  Episode {episode+1}: Exploit={current_exploit:.6f}, Reward={avg_reward:.4f}, Epsilon={agent.epsilon:.4f}")
                    
                    # Track best model
                    if current_exploit < best_exploitability:
                        best_exploitability = current_exploit
            
            # Save final strategy
            strategy_path = os.path.join('hyperparameter_test', f"{agent_id}_strategy.json")
            agent.save_strategy(strategy_path)
            
            # Store agent and metrics for comparison
            agents[agent_id] = agent
            exploit_histories[agent_id] = exploit_history
            reward_histories[agent_id] = reward_history
            
            print(f"  Final exploitability: {best_exploitability:.6f}")
            print(f"  Strategy saved to {strategy_path}")
    
    # Plot exploitability comparisons
    plt.figure(figsize=(12, 8))
    for agent_id, history in exploit_histories.items():
        iterations, exploits = zip(*history)
        epsilon, decay = agent_id.split('_')
        epsilon = epsilon[1:]  # Remove 'e' prefix
        decay = decay[1:]      # Remove 'd' prefix
        plt.plot(iterations, exploits, marker='o', label=f"ε={epsilon}, d={decay}")
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Exploitability')
    plt.title('Exploitability by Hyperparameter Configuration')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hyperparameter_test/exploitability_comparison.png')
    
    # Plot reward comparisons
    plt.figure(figsize=(12, 8))
    for agent_id, history in reward_histories.items():
        iterations, rewards = zip(*history)
        epsilon, decay = agent_id.split('_')
        epsilon = epsilon[1:]  # Remove 'e' prefix
        decay = decay[1:]      # Remove 'd' prefix
        plt.plot(iterations, rewards, marker='o', label=f"ε={epsilon}, d={decay}")
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Reward')
    plt.title('Average Reward by Hyperparameter Configuration')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('hyperparameter_test/reward_comparison.png')
    
    # Compare strategies against each other
    print("\nComparing trained strategies head-to-head:")
    
    # Get strategies
    strategies = {agent_id: agent.get_average_strategy() for agent_id, agent in agents.items()}
    
    # Compare head-to-head
    comparison = Evaluator.evaluate_strategies(strategies, num_games=5000)
    
    # Print results table
    print("\nHead-to-head results (row vs column, positive means row wins):")
    agent_ids = list(agents.keys())
    
    # Header
    header = "Agent ID".ljust(15)
    for col_id in agent_ids:
        header += col_id.ljust(15)
    print(header)
    
    # Table content
    for row_id in agent_ids:
        line = row_id.ljust(15)
        for col_id in agent_ids:
            if row_id == col_id:
                line += "---".ljust(15)
            else:
                value = comparison[row_id][col_id]
                line += f"{value:.4f}".ljust(15)
        print(line)
    
    # Save results
    results = {
        "exploitability": {agent_id: exploit_histories[agent_id][-1][1] for agent_id in agents},
        "head_to_head": comparison
    }
    
    Evaluator.save_results(results, os.path.join('hyperparameter_test', "comparison_results.json"))
    print("\nExperiment results saved to hyperparameter_test/")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    test_cfr_agent()
    
    # Run hyperparameter comparison experiment
    #compare_hyperparameters()
    
    # Uncomment to run self-play training (takes longer)
    #test_self_play_training(10000)