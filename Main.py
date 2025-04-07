import numpy as np
import os
from tqdm import tqdm
from config import Config
from royal_poker.cfr import KuhnCFR
from royal_poker.evaluate import Evaluator
from royal_poker.utils import save_strategy
import matplotlib.pyplot as plt

def main():
    results = {}
    strategies = {}
    
    # Create directories if they don't exist
    os.makedirs('strategy', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Run experiments with different configurations
    # Using only 3 buckets since we only have 3 distinct card values (0-Jack, 1-Queen, 2-King)
    for variant in ["Vanilla", "CFR+", "Linear"]:  # Added Linear CFR
        Config.CFR_VARIANT = variant
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
        strategies[experiment_name] = cfr.strategy
        save_strategy(cfr.strategy, os.path.join('strategy', f"{experiment_name}_strategy.json"))
        
        # Calculate final exploitability
        exploitability = Evaluator.calculate_exploitability(cfr.strategy)
        results[experiment_name] = {"exploitability": exploitability}
        
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
    
    # Compare strategies against each other
    comparison = Evaluator.evaluate_strategies(strategies, num_games=50000)  # More games for better estimates
    results["comparisons"] = comparison
    
    # Save all results
    Evaluator.save_results(results, os.path.join('results', "experiment_results.json"))
    print("Experiments completed. Results saved to results/experiment_results.json")

if __name__ == "__main__":
    main()