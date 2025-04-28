import numpy as np
import random
from royal_poker.utils import action_translation, load_strategy

class PokerAgent:
    """Base class for poker agents"""
    def __init__(self, name="Agent"):
        self.name = name
    
    def act(self, observation, legal_actions):
        """Take an action based on the current observation"""
        raise NotImplementedError("Subclasses must implement act method")

class CFRAgent(PokerAgent):
    """An agent that uses a CFR-trained strategy"""
    def __init__(self, strategy_path=None, name="CFR"):
        super().__init__(name)
        self.strategy = {}
        if strategy_path:
            self.load_strategy(strategy_path)
    
    def load_strategy(self, strategy_path):
        """Load a strategy from a file"""
        self.strategy = load_strategy(strategy_path)
        print(f"Loaded strategy with {len(self.strategy)} information sets")
        return self.strategy
    
    def act(self, observation, legal_actions):
        """Select an action based on the CFR strategy"""
        # Parse observation
        bucket, history = observation
        infostate = (bucket, tuple(history))
        
        # Check if we're responding to a bet
        is_response_to_bet = False
        if history and history[-1] == "bet":
            is_response_to_bet = True
        
        # Try to find strategy for this information state
        if infostate in self.strategy:
            probs = self.strategy[infostate]
            action_idx = np.random.choice(len(legal_actions), p=probs)
            abstract_action = ["check", "bet"][action_idx]
            
            # Translate abstract action to concrete action if necessary
            action = action_translation(abstract_action, is_response_to_bet)
            
            return action
            
        # If no exact match in strategy, try simplifying the history
        if len(history) > 0:
            simplified_state = (bucket, (history[-1],))
            if simplified_state in self.strategy:
                probs = self.strategy[simplified_state]
                action_idx = np.random.choice(len(legal_actions), p=probs)
                abstract_action = ["check", "bet"][action_idx]
                action = action_translation(abstract_action, is_response_to_bet)
                return action
        
        # Fallback to random if no strategy found
        return random.choice(legal_actions)

class RuleBasedAgent(PokerAgent):
    """A simple rule-based agent for Kuhn Poker"""
    def __init__(self, name="RuleBased"):
        super().__init__(name)
    
    def act(self, observation, legal_actions):
        """Select an action based on simple rules"""
        bucket, history = observation
        
        # Strategy based on card value (bucket)
        # In Kuhn poker: 0 = Jack, 1 = Queen, 2 = King
        
        # First action
        if not history:
            if bucket == 2:  # King
                return "bet" if random.random() < 0.7 else "check"
            elif bucket == 1:  # Queen
                return "bet" if random.random() < 0.3 else "check"
            else:  # Jack
                return "bet" if random.random() < 0.1 else "check"
        
        # Responding to check
        if history[-1] == "check":
            if bucket == 2:  # King
                return "bet" if random.random() < 0.8 else "check"
            elif bucket == 1:  # Queen
                return "bet" if random.random() < 0.5 else "check"
            else:  # Jack
                return "bet" if random.random() < 0.2 else "check"
        
        # Responding to bet
        if history[-1] == "bet":
            if bucket == 2:  # King
                return "call"  # Always call with King
            elif bucket == 1:  # Queen
                return "call" if random.random() < 0.7 else "fold"
            else:  # Jack
                return "call" if random.random() < 0.1 else "fold"
        
        # Default action (shouldn't reach here in Kuhn poker)
        return random.choice(legal_actions)
