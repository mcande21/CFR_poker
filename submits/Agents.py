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
        Update regrets and strategy based on observed reward.
        
        Args:
            observation: The state where the action was taken
            action: The action that was taken
            reward: The reward received
            next_observation: The resulting state (not used in CFR updates)
            terminal: Whether this was a terminal state (not used in CFR updates)
        """
        if not self.last_info_state:
            return
        
        # Convert concrete action to abstract action index
        is_response_to_bet = False
        bucket, history = observation
        if history and history[-1] == "bet":
            is_response_to_bet = True
        
        # Map the real action back to abstract action index
        if is_response_to_bet:
            action_idx = 1 if action == "call" else 0  # call->bet(1), fold->check(0)
        else:
            action_idx = 1 if action == "bet" else 0
        
        # Update regrets based on the received reward
        # This needs to be implemented further. For now we are just giving a reward
        # based on actions. The actual regret is still needed to be calculated
        # based off pot size.
        
        regrets = np.zeros(len(self.actions))
        for i in range(len(self.actions)):
            if i == action_idx:
                # No regret for the action actually taken
                regrets[i] = 0
            else:
                # This is a simplification; in full CFR we would compute proper counterfactual values
                regrets[i] = -reward  #we have negative regret for other actions
        
        # Update cumulative regrets
        self.regrets[observation] += regrets
        
        # Update strategy sum for computing average strategy
        self.strategy_sum[observation] += self.strategy[observation]
        
        # Increment iteration count
        self.iteration_count += 1
        
        # Decay exploration rate
        self.epsilon *= self.decay


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

