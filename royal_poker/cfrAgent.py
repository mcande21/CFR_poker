# Counterfactual Regret Minimization agent
from random import random, choice
import numpy as np
from collections import defaultdict
from royal_poker.utils import load_strategy, save_strategy

class CFRAgent(object):
    def __init__(self, env, strategy_path=None, epsilon=0.5, decay=0.99):
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
    
    def choose(self, observation, legal_actions):
        """
        Choose an action based on the current strategy with epsilon-exploration.
        
        Args:
            observation: Current game state (bucket, history)
            legal_actions: List of legal actions in the current state
        
        Returns:
            The selected action
        """
        # Store observation for later updates
        self.last_info_state = observation
        
        # Check if we're responding to a bet (this affects action meanings)
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
        
        # Get counterfactual values for all actions
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
    
    def _compute_counterfactual_values(self, bucket, history, is_terminal):
        """
        Compute counterfactual values for all actions using a forward simulation based on
        current strategy profiles.
        
        Args:
            bucket: Card bucket (0=Jack, 1=Queen, 2=King in Kuhn poker)
            history: Action history
            is_terminal: Whether we're at a terminal state
            
        Returns:
            Array of counterfactual values for each action
        """
        # Return values for each action
        cf_values = np.zeros(len(self.actions))
        
        # If we're at a terminal state, we can't take any more actions
        if is_terminal:
            return cf_values
        
        # Check if we're responding to a bet
        is_response_to_bet = False
        if history and history[-1] == "bet":
            is_response_to_bet = True
        
        # For each possible action
        for action_idx, abstract_action in enumerate(self.actions):
            # Map abstract action to concrete action
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
                # For Kuhn poker, we can use a simple recursive simulation
                # with our current strategy
                cf_values[action_idx] = self._simulate_to_end(sim_env, 10)  # depth limit of 10
        
        return cf_values
    
    def _simulate_to_end(self, env, depth_remaining):
        """
        Simulate playing to the end of the game using current strategy.
        
        Args:
            env: Environment to simulate in
            depth_remaining: Maximum depth to simulate (to prevent infinite loops)
            
        Returns:
            Expected value of the current position
        """
        if depth_remaining <= 0:
            # Emergency exit for unexpected infinite loops
            return 0
        
        # Get current state
        state = env.state()
        
        # If terminal, return the actual payoff
        if env.is_terminal():
            return env.get_payoff(env.current_player)
        
        # Get legal actions
        legal_actions = env.get_legal_actions()
        
        # Get strategy for this state
        strategy = self.get_strategy(state)
        
        # Calculate expected value by simulating each action
        expected_value = 0
        
        # For each action in our strategy
        for action_idx, action_prob in enumerate(strategy):
            if action_prob > 0:  # Only consider actions with non-zero probability
                # Map abstract action to concrete
                concrete_action = self._map_to_concrete_action(state, self.actions[action_idx])
                
                if concrete_action in legal_actions:
                    # Clone environment to avoid modifying original
                    sim_env = env.clone()
                    
                    # Take the action
                    _, _, done = sim_env.step(concrete_action)
                    
                    # Get the result, either terminal payoff or recursive simulation
                    if done:
                        action_value = sim_env.get_payoff(sim_env.current_player)
                    else:
                        action_value = self._simulate_to_end(sim_env, depth_remaining - 1)
                    
                    # Weight by probability in our strategy
                    expected_value += action_prob * action_value
        
        return expected_value
    
    def _map_to_concrete_action(self, state, abstract_action):
        """Map abstract action to concrete action based on game state"""
        bucket, history = state
        is_response_to_bet = history and history[-1] == "bet"
        
        if is_response_to_bet:
            return "call" if abstract_action == "bet" else "fold"
        else:
            return abstract_action
    
    def get_average_strategy(self):
        """
        Compute the average strategy across all iterations.
        In CFR, the average strategy converges to a Nash equilibrium.
        """
        avg_strategy = {}
        
        for info_state, strategy_sum in self.strategy_sum.items():
            total = np.sum(strategy_sum)
            if total > 0:
                avg_strategy[info_state] = strategy_sum / total
            else:
                # Uniform random as fallback
                avg_strategy[info_state] = np.ones(len(self.actions)) / len(self.actions)
        
        return avg_strategy
    
    def save_strategy(self, filename):
        """Save the current average strategy to a file"""
        save_strategy(self.get_average_strategy(), filename)
        print(f"Strategy saved to {filename}")
    
    def load_strategy(self, filename):
        """Load a pre-trained strategy from a file"""
        loaded_strategy = load_strategy(filename)
        
        # Convert loaded strategy to our format
        for info_state, probs in loaded_strategy.items():
            # Initialize regrets that would produce this strategy
            if np.sum(probs) > 0:
                # Set regrets proportional to the loaded probabilities
                self.regrets[info_state] = probs * 100  # Arbitrary scaling factor
                # Set strategy directly
                self.strategy[info_state] = probs
                # Add to strategy sum to influence average
                self.strategy_sum[info_state] = probs * 10  # Weight for average calculation
        
        print(f"Loaded strategy with {len(loaded_strategy)} states from {filename}")
        
        return self.strategy