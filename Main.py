import numpy as np
import os
from tqdm import tqdm
from config import Config
from royal_poker.cfr import KuhnCFR
from royal_poker.evaluate import Evaluator
from royal_poker.utils import save_strategy, load_strategy
from royal_poker.env import KuhnPokerEnv
from royal_poker import IncrementalCFRAgent
import matplotlib.pyplot as plt
import time

def train_standard_cfr(variant):
    """Train using standard CFR algorithm"""
    experiment_name = f"3B_{variant}"
    print(f"Running experiment: {experiment_name}")
    
    # Track convergence history for plotting
    exploit_history = []
    
    # Train CFR agent with enhanced monitoring
    cfr = KuhnCFR()
    
    # Implement early stopping if enabled
    best_exploitability = float('inf')
    patience_counter = 0
    
    # Train with periodic evaluation
    for i in range(0, Config.CFR_ITERATIONS, Config.EVAL_FREQUENCY):
        # Train for a batch of iterations
        batch_size = min(Config.EVAL_FREQUENCY, Config.CFR_ITERATIONS - i)
        
        # Temporarily override iterations for batch training
        original_iterations = Config.CFR_ITERATIONS
        Config.CFR_ITERATIONS = batch_size
        cfr.iteration_count = i
        cfr.train()
        Config.CFR_ITERATIONS = original_iterations
        
        # Evaluate current exploitability
        if Config.EARLY_STOPPING:
            current_exploit = Evaluator.calculate_exploitability(cfr.strategy)
            exploit_history.append((i + batch_size, current_exploit))
            
            print(f"Iteration {i + batch_size}: Exploitability = {current_exploit:.6f}")
            
            # Check for early stopping
            if current_exploit < best_exploitability - Config.CONVERGENCE_THRESHOLD:
                best_exploitability = current_exploit
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at iteration {i + batch_size}")
                break
    
    # Save the strategy
    strategy_path = os.path.join('strategy', f"{experiment_name}_strategy.json")
    save_strategy(cfr.strategy, strategy_path)
    
    # Calculate final exploitability
    exploitability = Evaluator.calculate_exploitability(cfr.strategy)
    result = {"exploitability": exploitability}
    
    # Plot convergence history if available
    if exploit_history:
        plt.figure(figsize=(10, 6))
        iterations, exploits = zip(*exploit_history)
        plt.plot(iterations, exploits)
        plt.xlabel('Iterations')
        plt.ylabel('Exploitability')
        plt.title(f'Convergence for {experiment_name}')
        plt.yscale('log')  # Log scale to better see improvements
        plt.savefig(f'logs/{experiment_name}_convergence.png')
        plt.close()
        
    return cfr.strategy, result

def train_incremental_cfr():
    """Train using incremental CFR agent"""
    experiment_name = "3B_Incremental"
    print(f"Running experiment: {experiment_name}")
    
    # Create environment
    env = KuhnPokerEnv()
    
    # Initialize incremental CFR agent
    agent = IncrementalCFRAgent(env=env, epsilon=0.5, decay=0.99)
    
    # Set up training parameters
    num_episodes = Config.CFR_ITERATIONS
    
    # Keep track of evaluation metrics
    total_reward = 0
    exploit_history = []
    eval_frequency = Config.EVAL_FREQUENCY
    best_exploitability = float('inf')
    patience_counter = 0
    
    # For progress tracking
    progress_bar = tqdm(range(num_episodes), desc="Incremental CFR Training")
    
    # Time tracking
    start_time = time.time()
    
    # Training loop
    for episode in progress_bar:
        # Reset environment
        env.reset()
        done = False
        
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
                
                total_reward += reward
        
        # Periodically evaluate and save
        if (episode + 1) % eval_frequency == 0:
            # Get average reward
            avg_reward = total_reward / (episode + 1)
            
            # Get current average strategy
            current_strategy = agent.get_average_strategy()
            
            # Evaluate exploitability
            current_exploit = Evaluator.calculate_exploitability(current_strategy)
            exploit_history.append((episode + 1, current_exploit))
            
            # Update progress info
            elapsed_time = time.time() - start_time
            progress_bar.set_postfix({
                "Exploit": f"{current_exploit:.6f}", 
                "Epsilon": f"{agent.epsilon:.4f}",
                "Time": f"{elapsed_time:.1f}s"
            })
            
            # Check for early stopping
            if Config.EARLY_STOPPING:
                if current_exploit < best_exploitability - Config.CONVERGENCE_THRESHOLD:
                    best_exploitability = current_exploit
                    patience_counter = 0
                    
                    # Save best strategy so far
                    tmp_path = os.path.join('strategy', f"{experiment_name}_best.json")
                    agent.save_strategy(tmp_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= Config.PATIENCE // eval_frequency:
                    print(f"Early stopping at episode {episode + 1}")
                    break
    
    # Save final strategy
    strategy_path = os.path.join('strategy', f"{experiment_name}_strategy.json")
    agent.save_strategy(strategy_path)
    
    # Plot convergence history
    if exploit_history:
        plt.figure(figsize=(10, 6))
        iterations, exploits = zip(*exploit_history)
        plt.plot(iterations, exploits)
        plt.xlabel('Episodes')
        plt.ylabel('Exploitability')
        plt.title(f'Convergence for {experiment_name}')
        plt.yscale('log')  # Log scale to better see improvements
        plt.savefig(f'logs/{experiment_name}_convergence.png')
        plt.close()
    
    # Return final strategy and results
    final_strategy = agent.get_average_strategy()
    result = {"exploitability": current_exploit}
    
    return final_strategy, result

def main():
    results = {}
    strategies = {}
    
    # Create directories if they don't exist
    os.makedirs('strategy', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run standard CFR experiments
    for variant in ["Vanilla", "CFR+", "Linear"]:
        Config.CFR_VARIANT = variant
        strategy, result = train_standard_cfr(variant)
        experiment_name = f"3B_{variant}"
        strategies[experiment_name] = strategy
        results[experiment_name] = result
    
    # Train incremental CFR agent
    incremental_strategy, incremental_result = train_incremental_cfr()
    strategies["3B_Incremental"] = incremental_strategy
    results["3B_Incremental"] = incremental_result
    
    # Compare strategies against each other
    print("Comparing all strategies against each other...")
    comparison = Evaluator.evaluate_strategies(strategies, num_games=50000)  # More games for better estimates
    results["comparisons"] = comparison
    
    # Save all results
    Evaluator.save_results(results, os.path.join('results', "experiment_results.json"))
    print("Experiments completed. Results saved to results/experiment_results.json")

if __name__ == "__main__":
    main()