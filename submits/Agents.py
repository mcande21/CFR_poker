# Counterfactual Regret Minimization Agent
from random import random, choice
import numpy as np
from collections import defaultdict
from utils import save_strategy, load_strategy

class CFRAgent(object):
    def __init__(self, env, strategy_path=None, epsilon=0.3, decay=0.999):
        """
        CFR-based agent
        
        Args:
            env: environment
            strategy_path: Path to pre-trained strategy file
            epsilon: Exploration rate for non-optimal actions
            decay: Rate at which epsilon decays over time
        """
        self.env = env
        self.epsilon = epsilon
        self.decay = decay
        self.actions = ["check", "bet"]  # Abstract action space for Kuhn Poker
        
        # Regret tracking
        self.regrets = defaultdict(lambda: np.zeros(len(self.actions)))
        # Strategy sums for averaging
        self.strategy_sum = defaultdict(lambda: np.zeros(len(self.actions)))
        # Initialize current strategy
        self.strategy = defaultdict(lambda: np.ones(len(self.actions))/len(self.actions))
        
        # Training iteration tracking
        self.iteration_count = 0
        self.last_info_state = None  # For storage
        
        # Load pre-trained strategy if provided
        if strategy_path:
            self.load_strategy(strategy_path)

    def load_strategy(self, strategy_path):
        """
        Load a pre-trained strategy from a file.
        
        Args:
            strategy_path: Path to the strategy file
        """
        try:
            loaded_data = load_strategy(strategy_path)
            # Convert string keys back to tuples for info states
            for key_str, value in loaded_data.items():
                # Parse string representation of tuple like "(0, ('check',))"
                # This is a simplification and may need adjustment based on your exact format
                key = eval(key_str)
                self.strategy[key] = np.array(value)
                # Initialize strategy sums with the loaded strategy
                self.strategy_sum[key] = self.strategy[key] * self.iteration_count
            print(f"Successfully loaded strategy from {strategy_path}")
        except Exception as e:
            print(f"Error loading strategy from {strategy_path}: {e}")
            # Continue with default uniform strategy
    
    def choose(self, observation, legal_actions):
        """
        Choose an action based on the current strategy with epsilon-exploration.
        
        Args:
            observation: Current game state (card, history)
            legal_actions: List of legal actions in the current state
        
        Returns:
            The selected action
        """
        # Store observation for later updates
        self.last_info_state = observation
        
        # Check if we're responding to a bet (action is then call/fold)
        is_response_to_bet = False
        bucket, history = observation
        if history and history[-1] == "bet":
            is_response_to_bet = True
        
        # Get strategy for current state
        strategy = self.get_strategy(observation)
        
        # Epsilon-greedy exploration
        if random() < self.epsilon:
            # Random action
            action_idx = choice(range(len(self.actions)))
        else:
            # Sample from strategy distribution
            action_idx = np.random.choice(len(self.actions), p=strategy)
        
        # Translate abstract action to concrete action if necessary
        abstract_action = self.actions[action_idx]
        if is_response_to_bet:
            # When responding to a bet, "check" means "fold" and "bet" means "call"
            if abstract_action == "check":
                return "fold"
            else:  # abstract_action == "bet"
                return "call"
        else:
            return abstract_action
    
    def update(self, observation, action, reward, next_observation=None, terminal=False):
        """
        Update regrets using proper counterfactual value calculations.
        
        Args:
            observation: The state where the action was taken
            action: The action that was taken
            reward: The reward received
            next_observation: The resulting state
            terminal: Whether this was a terminal state
        """
        if not self.last_info_state:
            return
        
        # Store game state information
        bucket, history = observation
        
        # For actions that make terminal states, we can use the actual reward
        if terminal:
            # Simple case: direct update with the given reward
            # Map the real action back to abstract action index
            is_response_to_bet = False
            if history and history[-1] == "bet":
                is_response_to_bet = True
            
            if is_response_to_bet:
                action_idx = 1 if action == "call" else 0  # call->bet(1), fold->check(0)
            else:
                action_idx = 1 if action == "bet" else 0
                
            # Create regrets assuming all other actions would have had 0 reward
            regrets = np.zeros(len(self.actions))
            for i in range(len(self.actions)):
                if i == action_idx:
                    # No regret for the action we took
                    regrets[i] = 0
                else:
                    # For testing purposes, assume alternative actions would have had opposite reward
                    # In a real game, this would be calculated via counterfactual values
                    regrets[i] = -reward  # If reward was negative, positive regret for not taking this action
            
            # Apply the regrets directly
            self.regrets[observation] = np.maximum(self.regrets[observation] + regrets, 0)
            
            # Update strategy sum
            weight = np.sqrt(self.iteration_count + 1)
            self.strategy_sum[observation] += weight * self.strategy[observation]
            
            # Increment iteration count and decay epsilon
            self.iteration_count += 1
            self.epsilon *= self.decay
            return
        
        # If not terminal, use counterfactual values
        cf_values = self._compute_counterfactual_values(bucket, history, terminal)
        
        # Get the utility of the action we actually took
        is_response_to_bet = False
        if history and history[-1] == "bet":
            is_response_to_bet = True
        
        # Map the real action back to abstract action index
        if is_response_to_bet:
            action_idx = 1 if action == "call" else 0  # call->bet(1), fold->check(0)
        else:
            action_idx = 1 if action == "bet" else 0
        
        # The reward we actually received becomes our baseline
        actual_value = reward
        
        # Calculate regrets: difference between counterfactual value and actual value
        regrets = np.zeros(len(self.actions))
        for i in range(len(self.actions)):
            if i == action_idx:
                # No regret for action actually taken (could be slightly negative due to sampling)
                regrets[i] = 0
            else:
                # Regret is the difference between what we could have gotten and what we got
                regrets[i] = cf_values[i] - actual_value
        
        # Calculate pot size for scaling regrets
        pot_size = 2 + history.count("bet")  # Base pot (2) + number of bets
        
        # Scale regrets by square root of pot size for balanced updates
        regrets = regrets * np.sqrt(pot_size)
        
        # Apply CFR+ techniques for faster convergence
        # 1. Regret Matching+: only allow positive regrets to accumulate
        self.regrets[observation] = np.maximum(self.regrets[observation] + regrets, 0)
        
        # 2. Weighted averaging: give more weight to later iterations
        # Calculate weight factor based on iteration (increases importance of later iterations)
        weight = np.sqrt(self.iteration_count + 1)
        self.strategy_sum[observation] += weight * self.strategy[observation]
        
        # Increment iteration count
        self.iteration_count += 1
        
        # Decay exploration rate
        self.epsilon *= self.decay

    def get_strategy(self, info_state):
        """
        Get strategy for an information state using regret-matching.
        The strategy probability for each action is proportional to its positive regret.
        """
        regrets = self.regrets[info_state]
        strategy = np.zeros(len(self.actions))
        
        # Only consider positive regrets (negative regrets mean we should avoid these actions)
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            # Normalize strategy based on positive regrets
            strategy = positive_regrets / regret_sum
        else:
            # If no positive regrets, use uniform strategy
            strategy = np.ones(len(self.actions)) / len(self.actions)
        
        # Store in current strategy
        self.strategy[info_state] = strategy
        
        return strategy

    def get_average_strategy(self):
        """
        Compute the average strategy across all training iterations.
        This average strategy converges to a Nash equilibrium.
        
        Returns:
            Dict mapping information states to probability distributions over actions
        """
        avg_strategy = {}
        
        for info_state, strategy_sum in self.strategy_sum.items():
            # Calculate the average strategy for this information state
            total = np.sum(strategy_sum)
            if total > 0:
                # Normalize to get probabilities
                avg_strategy[info_state] = strategy_sum / total
            else:
                # If we haven't visited this state much, use a uniform strategy
                avg_strategy[info_state] = np.ones(len(self.actions)) / len(self.actions)
        
        return avg_strategy
    
    def save_strategy(self, strategy_path):
        """
        Save the current average strategy to a file.
        
        Args:
            strategy_path: Path where to save the strategy
        """
        # Get the average strategy
        strategy_to_save = self.get_average_strategy()
        
        # Convert to a dictionary with string keys (for JSON serialization)
        serializable_strategy = {}
        for info_state, probs in strategy_to_save.items():
            # Convert tuple to a string representation
            key_str = str(info_state)
            serializable_strategy[key_str] = probs.tolist()
        
        # Save to file
        save_strategy(serializable_strategy, strategy_path)
        print(f"Strategy saved to {strategy_path}")

    def _compute_counterfactual_values(self, bucket, history, is_terminal):
        """
        Compute counterfactual values for all actions in the given state.
        This is a simplified version for Kuhn Poker.
        
        Args:
            bucket: The card held by the player (0=Jack, 1=Queen, 2=King)
            history: Action history
            is_terminal: Whether this is a terminal state
            
        Returns:
            Numpy array of counterfactual values for each action
        """
        cf_values = np.zeros(len(self.actions))
        
        # For each possible action
        for action_idx, abstract_action in enumerate(self.actions):
            # Map abstract to concrete action
            is_response_to_bet = False
            if history and history[-1] == "bet":
                is_response_to_bet = True
            
            if is_response_to_bet:
                concrete_action = "call" if abstract_action == "bet" else "fold"
            else:
                concrete_action = abstract_action  # "check" or "bet"
            
            # Create a copy of the environment to simulate this action
            sim_env = self.env.clone()
            
            # Set the state to match our current observation
            sim_env.set_state(bucket, history)
            
            # Take the action
            _, _, sim_done = sim_env.step(concrete_action)
            
            # If game is over, get the payoff
            if sim_done:
                # Get the payoff from the simulated environment
                cf_values[action_idx] = sim_env.get_payoff(sim_env.current_player)
            else:
                # If game isn't over, we need to estimate the expected value
                # For Kuhn poker, we can use a simple rollout with current strategy
                next_state = sim_env.state()
                legal_actions = sim_env.get_legal_actions()
                
                # Use current strategy to choose next action
                next_strategy = self.get_strategy(next_state)
                next_action_idx = np.random.choice(len(self.actions), p=next_strategy)
                next_abstract_action = self.actions[next_action_idx]
                
                # Map to concrete action
                next_state_responding_to_bet = False
                _, next_history = next_state
                if next_history and next_history[-1] == "bet":
                    next_state_responding_to_bet = True
                
                if next_state_responding_to_bet:
                    next_concrete_action = "call" if next_abstract_action == "bet" else "fold"
                else:
                    next_concrete_action = next_abstract_action
                
                # Take action if it's legal
                if next_concrete_action in legal_actions:
                    _, _, final_done = sim_env.step(next_concrete_action)
                    # Get payoff
                    if final_done:
                        cf_values[action_idx] = sim_env.get_payoff(sim_env.current_player)
                    else:
                        # For Kuhn poker, we shouldn't reach here (max 2 actions per player)
                        cf_values[action_idx] = 0
                else:
                    # If action not legal, use default
                    cf_values[action_idx] = 0
                
        return cf_values

class RuleBasedAgent(object):
    """A simple rule-based agent for Kuhn Poker"""
    def __init__(self, name="RuleBased"):
        self.name = name
    
    def choose(self, observation, legal_actions):
        """Select an action based on simple rules"""
        bucket, history = observation
        
        # Strategy based on card value
        # In Kuhn poker: 0 = Jack, 1 = Queen, 2 = King
        
        # First action
        if not history:
            if bucket == 2:  # King
                return "bet" if random() < 0.7 else "check"
            elif bucket == 1:  # Queen
                return "bet" if random() < 0.3 else "check"
            else:  # Jack
                return "bet" if random() < 0.1 else "check"
        
        # Responding to check
        if history[-1] == "check":
            if bucket == 2:  # King
                return "bet" if random() < 0.8 else "check"
            elif bucket == 1:  # Queen
                return "bet" if random() < 0.5 else "check"
            else:  # Jack
                return "bet" if random() < 0.2 else "check"
        
        # Responding to bet
        if history[-1] == "bet":
            if bucket == 2:  # King
                return "call"  # Always call with King
            elif bucket == 1:  # Queen
                return "call" if random() < 0.7 else "fold"
            else:  # Jack
                return "call" if random() < 0.1 else "fold"
        
        # Default action (shouldn't reach here in Kuhn poker)
        return choice(legal_actions)
    
    # Adding update method for compatibility with CFR agent interface
    def update(self, observation, action, reward, next_observation=None, terminal=False):
        """
        Rule-based agents don't learn, but implementing this method for interface compatibility
        """
        pass

