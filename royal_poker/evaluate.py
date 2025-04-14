import numpy as np
import json
import os
from royal_poker.env import KuhnPokerEnv
from config import Config

class Evaluator:
    @staticmethod
    def calculate_exploitability(strategy):
        """
        Calculate how much a strategy can be exploited using a best response calculation.
        
        Exploitability is a measure of how far a strategy is from a Nash equilibrium.
        A Nash equilibrium has zero exploitability, meaning it cannot be beaten in expectation.
        This method computes the worst-case expected loss against a perfect opponent.
        
        Args:
            strategy: The strategy to evaluate (a dictionary mapping info states to action probabilities)
            
        Returns:
            float: The exploitability value (lower is better, zero is perfect)
        """
        env = KuhnPokerEnv()
        exploitability = 0.0
        
        # Create best response strategies for each player
        best_response_value = 0
        
        # For each card the opponent can have
        for opponent_card in range(Config.DECK_SIZE):
            # Calculate expected value against all possible player's cards
            ev = 0
            valid_combinations = 0
            
            for player_card in range(Config.DECK_SIZE):
                if player_card == opponent_card:  # Skip impossible hands
                    continue
                    
                # Initialize environment with fixed cards
                env.reset()
                env.hands = [player_card, opponent_card]
                
                # Compute the best response value - this is the maximum value an opponent
                # can achieve against our strategy in this situation
                value = Evaluator._get_best_response_value(env, strategy, 0)
                ev += value
                valid_combinations += 1
            
            # Average EV against this opponent card
            if valid_combinations > 0:
                best_response_value += ev / valid_combinations
        
        # Average over all opponent cards - this is our exploitability
        exploitability = best_response_value / Config.DECK_SIZE
        
        return 1 - abs(exploitability)
    
    @staticmethod
    def _get_best_response_value(env, strategy, depth=0):
        """
        Recursive calculation of best response value.
        
        This computes how much value an optimal opponent can extract when:
        - Our player uses the given strategy
        - The opponent makes perfect decisions to maximize their value
        
        Args:
            env: The game environment
            strategy: The strategy to evaluate
            depth: Current recursion depth
            
        Returns:
            float: Expected value for the opponent (maximized)
        """
        if env.terminal:
            return env.get_payoff(1)  # Return opponent's payoff
        
        player = env.current_player
        infostate = env.state()
        actions = env.get_legal_actions()
        
        # Current player uses the strategy
        if player == 0:
            # Get probabilities from strategy
            probs = strategy.get(infostate, np.ones(2)/2)
            
            # Calculate expected value
            expected_value = 0
            
            # Save current game state
            history = env.history.copy()
            terminal = env.terminal
            hands = env.hands.copy()
            
            for i, action in enumerate(actions):
                # Restore game state
                env.history = history.copy()
                env.terminal = terminal
                env.hands = hands.copy()
                
                # Take action
                env.step(action)
                
                # Recursive call
                action_value = Evaluator._get_best_response_value(env, strategy, depth+1)
                expected_value += probs[i] * action_value
            
            return expected_value
        else:
            # Opponent chooses best response (max value)
            best_value = -float('inf')
            
            # Save current game state
            history = env.history.copy()
            terminal = env.terminal
            hands = env.hands.copy()
            
            for action in actions:
                # Restore game state
                env.history = history.copy()
                env.terminal = terminal
                env.hands = hands.copy()
                
                # Take action
                env.step(action)
                
                # Recursive call
                action_value = Evaluator._get_best_response_value(env, strategy, depth+1)
                best_value = max(best_value, action_value)
            
            return best_value

    @staticmethod
    def save_results(results, filename="results.json"):
        # Make sure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
            
    @staticmethod
    def evaluate_strategies(strategies, num_games=1000):
        """Compare different strategies by playing them against each other"""
        results = {}
        env = KuhnPokerEnv()
        
        for name1, strategy1 in strategies.items():
            results[name1] = {}
            for name2, strategy2 in strategies.items():
                if name1 == name2:
                    continue
                    
                total_payoff = 0
                for _ in range(num_games):
                    env.reset()
                    while not env.terminal:
                        player = env.current_player
                        infostate = env.state()
                        actions = env.get_legal_actions()
                        
                        if player == 0:
                            # Get strategy probabilities or use uniform if not found
                            probs = strategy1.get(infostate, np.ones(len(actions))/len(actions))
                            # Ensure probabilities match the number of legal actions
                            if len(probs) != len(actions):
                                probs = np.ones(len(actions))/len(actions)
                        else:
                            # Get strategy probabilities or use uniform if not found
                            probs = strategy2.get(infostate, np.ones(len(actions))/len(actions))
                            # Ensure probabilities match the number of legal actions
                            if len(probs) != len(actions):
                                probs = np.ones(len(actions))/len(actions)
                        
                        # Normalize probabilities to ensure they sum to 1
                        probs = probs / np.sum(probs)
                        action = np.random.choice(actions, p=probs)
                        env.step(action)
                        
                    total_payoff += env.get_payoff(0)
                    
                results[name1][name2] = total_payoff / num_games
                
        return results
