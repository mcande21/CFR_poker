# Trainined CFR Agent vs Rule Based Agent. Win rate + Payout + Exploitability
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import json

from environment import KuhnPokerEnv
from Agents import CFRAgent, RuleBasedAgent
from utils import save_strategy

# Ensure strategies directory exists
if not os.path.exists('submits/strategies'):
    os.makedirs('submits/strategies')

def train_against_rule_based(num_iterations=50000, eval_frequency=5000, epsilon=0.3, decay=0.999):
    """
    Train the CFR agent against a rule-based agent and evaluate performance periodically.
    
    Args:
        num_iterations: Number of training iterations
        eval_frequency: How often to evaluate
        epsilon: Initial exploration rate for CFR agent
        decay: Decay rate for exploration
    """
    print(f"Starting training for {num_iterations} iterations...")
    print(f"CFR parameters: epsilon={epsilon}, decay={decay}")
    
    # Create the environment
    env = KuhnPokerEnv()
    
    # Create the agents
    cfr_agent = CFRAgent(env=env, epsilon=epsilon, decay=decay)
    rule_agent = RuleBasedAgent()
    
    # Training statistics
    win_rates = []
    tie_rates = []
    loss_rates = []
    payouts = []
    exploitability = []
    iterations = []
    
    training_start = time.time()
    
    # Main training loop
    for i in range(1, num_iterations + 1):
        # Reset environment
        state, done = env.reset(), False
        
        # Play one full game
        while not done:
            # Get legal actions for current player
            legal_actions = env.get_legal_actions()
            
            # Choose action based on current player
            if env.current_player == 0:  # CFR agent's turn
                action = cfr_agent.choose(state, legal_actions)
            else:  # Rule agent's turn
                action = rule_agent.choose(state, legal_actions)
            
            # Take action in environment
            next_state, reward, done = env.step(action)
            
            # Update CFR agent only if it was their turn
            if env.current_player == 0:
                cfr_agent.update(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
        
        # Logging and evaluation
        if i % 1000 == 0:
            elapsed = time.time() - training_start
            print(f"Iteration {i}/{num_iterations} ({i/num_iterations*100:.1f}%) - Time elapsed: {elapsed:.2f}s")
        
        # Periodic evaluation
        if i % eval_frequency == 0 or i == num_iterations:
            # Evaluate against rule-based agent
            win_rate, tie_rate, loss_rate, avg_payout = evaluate_against_rule(cfr_agent, rule_agent, env, num_eval_games=1000)
            win_rates.append(win_rate)
            tie_rates.append(tie_rate)
            loss_rates.append(loss_rate)
            payouts.append(avg_payout)
            
            # Estimate exploitability (simplified)
            current_exploit = estimate_exploitability(cfr_agent, env, num_samples=1000)
            exploitability.append(current_exploit)
            iterations.append(i)
            
            print(f"Iteration {i}: Win rate: {win_rate:.3f}, Tie rate: {tie_rate:.3f}, Loss rate: {loss_rate:.3f}, Avg payout: {avg_payout:.3f}, Exploitability: {current_exploit:.5f}")
            
            # Save statistics
            stats = {
                'iterations': iterations,
                'win_rates': win_rates,
                'tie_rates': tie_rates,
                'loss_rates': loss_rates,
                'payouts': payouts,
                'exploitability': exploitability,
                'params': {
                    'epsilon': epsilon,
                    'decay': decay
                }
            }
            
            with open('submits/strategies/training_stats.json', 'w') as f:
                json.dump(stats, f)
    
    # Final evaluation
    final_win_rate, final_tie_rate, final_loss_rate, final_payout = evaluate_against_rule(cfr_agent, rule_agent, env, num_eval_games=5000)
    
    print(f"\nTraining completed!")
    print(f"Final win rate against rule-based agent: {final_win_rate:.3f}")
    print(f"Final tie rate: {final_tie_rate:.3f}")
    print(f"Final loss rate: {final_loss_rate:.3f}")
    print(f"Final average payout: {final_payout:.3f}")
    
    # Save final strategy using utility function
    final_strategy_path = "submits/strategies/final_strategy.json"
    save_strategy(cfr_agent.get_average_strategy(), final_strategy_path)
    print(f"Final strategy saved to {final_strategy_path}")
    
    # Create and save plots
    plot_training_results(iterations, win_rates, tie_rates, loss_rates, payouts, exploitability)
    
    return cfr_agent

def evaluate_against_rule(cfr_agent, rule_agent, env, num_eval_games=1000):
    """
    Evaluate CFR agent performance against rule-based agent.
    
    Args:
        cfr_agent: Trained CFR agent
        rule_agent: Rule-based agent
        env: Kuhn poker environment
        num_eval_games: Number of games to evaluate
        
    Returns:
        (win_rate, tie_rate, loss_rate, average_payout)
    """
    wins = 0
    ties = 0
    losses = 0
    total_payouts = 0
    
    # Turn off exploration during evaluation
    original_epsilon = cfr_agent.epsilon
    cfr_agent.epsilon = 0
    
    for _ in range(num_eval_games):
        state, done = env.reset(), False
        
        # Play one full game
        while not done:
            legal_actions = env.get_legal_actions()
            
            if env.current_player == 0:  # CFR agent's turn
                action = cfr_agent.choose(state, legal_actions)
            else:  # Rule agent's turn
                action = rule_agent.choose(state, legal_actions)
            
            state, reward, done = env.step(action)
        
        # Get final payoff for CFR agent (player 0)
        final_payoff = env.get_payoff(0)
        total_payouts += final_payoff
        
        if final_payoff > 0:
            wins += 1
        elif final_payoff < 0:
            losses += 1
        else:
            ties += 1
    
    # Restore exploration rate
    cfr_agent.epsilon = original_epsilon
    
    win_rate = wins / num_eval_games
    tie_rate = ties / num_eval_games
    loss_rate = losses / num_eval_games
    
    return win_rate, tie_rate, loss_rate, total_payouts / num_eval_games

def estimate_exploitability(cfr_agent, env, num_samples=1000):
    """
    Estimate exploitability of the current strategy by calculating how much a best
    response player could gain against the CFR agent's strategy.
    
    Args:
        cfr_agent: The CFR agent
        env: Kuhn poker environment
        num_samples: Number of game simulations
        
    Returns:
        Estimated exploitability value
    """
    # Use CFR agent's average strategy
    strategy = cfr_agent.get_average_strategy()
    
    # Calculate exploitability for each starting card (0=Jack, 1=Queen, 2=King)
    total_exploitability = 0
    card_counts = {0: 0, 1: 0, 2: 0}  # Track number of each card dealt
    best_response_values = {0: 0, 1: 0, 2: 0}  # Value of best response per card
    
    # Run multiple simulations
    for _ in range(num_samples):
        # Reset the environment to get a fresh game state
        env.reset()
        
        # Get the initial card
        card = env.hands[0]  # Get player 0's card
        card_counts[card] += 1
        
        # Simulate best response for this card
        br_value = simulate_best_response(card, env, cfr_agent)
        best_response_values[card] += br_value
    
    # Calculate average best response value for each card
    for card in range(3):  # 0=Jack, 1=Queen, 2=King
        if card_counts[card] > 0:
            avg_br_value = best_response_values[card] / card_counts[card]
            # Weight by the probability of getting each card (approximately 1/3 for each)
            total_exploitability += avg_br_value / 3
    
    return max(0.0, total_exploitability)

def simulate_best_response(card, env, cfr_agent):
    """
    Simulates a best response player against the CFR agent for a specific starting card.
    
    Args:
        card: The starting card (0=Jack, 1=Queen, 2=King)
        env: Kuhn poker environment
        cfr_agent: The CFR agent to play against
    
    Returns:
        Expected value of the best response
    """
    # For Kuhn poker, we can explicitly check the best response by trying both actions
    # and seeing which gives better results
    best_value = float('-inf')
    
    # Simulate both initial actions: check and bet
    for first_action in ["check", "bet"]:
        # Clone the environment to avoid modifying original game state
        sim_env = env.clone()
        sim_env.set_state(card, tuple())  # Empty history
        
        # Take first action
        next_state, _, done = sim_env.step(first_action)
        
        if done:
            # Game ended immediately (shouldn't happen in Kuhn poker)
            value = sim_env.get_payoff(0)
        else:
            # Now it's opponent's turn - opponent follows the CFR strategy
            # Get opponent's action using CFR agent's strategy
            opponent_action = simulate_opponent_action(cfr_agent, sim_env.state())
            
            # Take opponent's action
            next_state, _, done = sim_env.step(opponent_action)
            
            if done:
                # Game ended after opponent's action
                value = sim_env.get_payoff(0)
            else:
                # If the game is still going, we need a second action
                # For Kuhn poker, this only happens when both check or when opponent bets
                if opponent_action == "bet":
                    # Opponent bet, we can call or fold - pick the best
                    call_env = sim_env.clone()
                    call_env.step("call")
                    call_value = call_env.get_payoff(0)
                    
                    fold_env = sim_env.clone()
                    fold_env.step("fold")
                    fold_value = fold_env.get_payoff(0)
                    
                    # Best response is to pick the maximum value action
                    value = max(call_value, fold_value)
                else:  # opponent_action == "check"
                    # Both checked, we can bet or check - pick the best
                    check_env = sim_env.clone()
                    check_env.step("check")
                    check_value = check_env.get_payoff(0)
                    
                    bet_env = sim_env.clone()
                    bet_env.step("bet")
                    # After we bet, opponent can call or fold
                    opponent_second_action = simulate_opponent_action(cfr_agent, bet_env.state())
                    bet_env.step(opponent_second_action)
                    bet_value = bet_env.get_payoff(0)
                    
                    # Best response is to pick the maximum value action
                    value = max(check_value, bet_value)
        
        # Update best value
        best_value = max(best_value, value)
    
    return best_value

def simulate_opponent_action(cfr_agent, state):
    """
    Simulates opponent's action selection using the CFR agent's strategy.
    
    Args:
        cfr_agent: The CFR agent (whose strategy we're using for opponent)
        state: Current game state
    
    Returns:
        Opponent's action selected according to CFR strategy
    """
    # Get the strategy for this state
    bucket, history = state
    strategy = cfr_agent.get_strategy(state)
    
    # Sample action from strategy distribution
    action_idx = np.random.choice(len(cfr_agent.actions), p=strategy)
    abstract_action = cfr_agent.actions[action_idx]
    
    # Map abstract action to concrete action
    is_response_to_bet = history and history[-1] == "bet"
    if is_response_to_bet:
        if abstract_action == "check":
            return "fold"
        else:  # abstract_action == "bet"
            return "call"
    else:
        return abstract_action

def plot_training_results(iterations, win_rates, tie_rates, loss_rates, payouts, exploitability):
    """Create and save plots of training metrics"""
    # Create figure for all plots
    plt.figure(figsize=(15, 15))
    
    # Game metrics plot (win, tie, loss rates)
    plt.subplot(3, 1, 1)
    plt.plot(iterations, win_rates, marker='o', label='Win Rate', color='green')
    plt.plot(iterations, tie_rates, marker='s', label='Tie Rate', color='blue')
    plt.plot(iterations, loss_rates, marker='^', label='Loss Rate', color='red')
    plt.title('Game Metrics vs. Rule-Based Agent')
    plt.xlabel('Training Iterations')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    
    # Average payout plot
    plt.subplot(3, 1, 2)
    plt.plot(iterations, payouts, marker='o', color='purple')
    plt.title('Average Payout vs. Rule-Based Agent')
    plt.xlabel('Training Iterations')
    plt.ylabel('Average Payout')
    plt.grid(True)
    
    # Exploitability plot
    plt.subplot(3, 1, 3)
    plt.plot(iterations, exploitability, marker='o', color='orange')
    plt.title('Estimated Exploitability')
    plt.xlabel('Training Iterations')
    plt.ylabel('Exploitability')
    plt.yscale('log')  # Log scale for exploitability
    plt.grid(True)
    
    # Save the figure with all plots
    plt.tight_layout()
    plt.savefig('submits/strategies/training_results.png')
    
    # Create a separate figure just for game metrics with a clearer view
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, win_rates, marker='o', label='Win Rate', color='green', linewidth=2)
    plt.plot(iterations, tie_rates, marker='s', label='Tie Rate', color='blue', linewidth=2)
    plt.plot(iterations, loss_rates, marker='^', label='Loss Rate', color='red', linewidth=2)
    plt.title('Game Metrics vs. Rule-Based Agent')
    plt.xlabel('Training Iterations')
    plt.ylabel('Rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('submits/strategies/game_metrics.png')
    
    print("Training results visualized and saved to submits/strategies/training_results.png")
    print("Game metrics visualized and saved to submits/strategies/game_metrics.png")

if __name__ == "__main__":
    # Train CFR agent against rule-based agent
    # We have been messing with parameters and they are quite volitile on the graphs because
    # the values are so small
    train_against_rule_based(
        num_iterations=40000,  # Total training iterations
        eval_frequency=4000,   # How often to evaluate performance
        epsilon=0.2,           # Initial exploration rate
        decay=0.990            # Decay rate for exploration
    )