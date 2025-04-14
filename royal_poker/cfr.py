import numpy as np
import random
from collections import defaultdict
from tqdm import tqdm
from config import Config
from royal_poker.env import KuhnPokerEnv
from royal_poker.utils import save_strategy, load_strategy

class KuhnCFR:
    """
    Counterfactual Regret Minimization implementation for Kuhn poker.
    
    CFR is a self-play algorithm that finds an approximate Nash equilibrium by:
    1. Starting with a uniformly random strategy
    2. Repeatedly playing against itself
    3. Calculating regret for actions not taken (how much better would it have been?)
    4. Updating the strategy to favor actions with positive regret
    5. Computing the average strategy over all iterations

    Supports several CFR variants: Vanilla CFR, CFR+, and Linear CFR.
    """
    
    def __init__(self):
        """Initialize the CFR solver with uniform random strategy"""
        self.env = KuhnPokerEnv()
        # Initialize regrets for [check, bet]
        self.regrets = defaultdict(lambda: np.zeros(2))
        # Accumulated strategy for averaging
        self.strategy_sum = defaultdict(lambda: np.zeros(2))
        self.iteration_count = 0
        
        # For tracking convergence
        self.history = []
        
    def get_strategy(self, infostate):
        """
        Get strategy for an information state using regret-matching.
        The strategy probability for each action is proportional to its positive regret.
        """
        # Get regrets for this information state
        regrets = self.regrets[infostate]
        
        # Apply regret matching to compute strategy
        strategy = np.zeros(2)
        
        # Only consider positive regrets (negative regrets mean we should avoid these actions)
        positive_regrets = np.maximum(regrets, 0)
        regret_sum = np.sum(positive_regrets)
        
        if regret_sum > 0:
            # Normalize strategy based on positive regrets
            # This makes the probability of taking an action proportional to its positive regret
            strategy = positive_regrets / regret_sum
        else:
            # If no positive regrets, use uniform strategy
            # This is the starting point for our learning process
            strategy = np.ones(2) / 2
            
        return strategy
    
    def compute_cfr(self, cards, history, pr_1, pr_2, reach_pr=1.0):
        """
        Compute counterfactual regrets and update strategy for both players.
        
        Args:
            cards: List of cards for each player (Jack=0, Queen=1, King=2)
            history: Game action history
            pr_1: Reach probability for player 1
            pr_2: Reach probability for player 2
            reach_pr: Total reach probability (for CFR variants)
        
        Returns:
            Expected utility for the current player
        """
        # Check if we're at a terminal state
        if len(history) >= 2:
            if (history[-2:] == ["check", "check"] or 
                history[-2:] == ["bet", "bet"] or
                history[-2:] == ["check", "bet", "bet"]):
                # Showdown
                self.env.hands = cards.copy()
                self.env.history = history.copy()
                self.env.determine_win()
                return self.env.get_payoff(0)
            elif history[-2:] == ["bet", "check"]:
                # Player folded
                return -1  # First player pays ante, second player wins
            elif history[-2:] == ["check", "bet", "check"]:
                # Player folded after a bet
                return 1  # Second player pays ante, first player wins
        
        # Determine current player (alternates with each action)
        player = len(history) % 2
        
        # If it's player 0's turn
        if player == 0:
            # Determine bucket for information set
            # For Kuhn poker with 3 cards, each card is its own bucket
            # No need to divide by (Config.DECK_SIZE // Config.BUCKETS) when BUCKETS == DECK_SIZE
            if Config.BUCKETS >= Config.DECK_SIZE:
                bucket = cards[player]  # Jack=0, Queen=1, King=2
            else:
                bucket = cards[player] // (Config.DECK_SIZE // Config.BUCKETS)
            
            # Create information set representation
            infostate = (bucket, tuple(history))
            
            # Get current strategy for this information state
            strategy = self.get_strategy(infostate)
            
            # Initialize expected value array
            action_utilities = np.zeros(2)
            
            # Compute expected value for each action
            for i, action in enumerate(["check", "bet"]):
                # Recursively compute utility for this action
                action_history = history + [action]
                
                # Calculate new reach probabilities
                new_pr_1 = pr_1 * strategy[i]
                
                # Recursive call
                action_utilities[i] = self.compute_cfr(cards, action_history, new_pr_1, pr_2, reach_pr)
            
            # Compute expected value given current strategy
            utility = np.sum(strategy * action_utilities)
            
            # Only update regrets and strategy if CFR iterations haven't reached max
            if self.iteration_count < Config.CFR_ITERATIONS:
                # Compute counterfactual reach probability
                counterfactual_reach_prob = pr_2
                
                # Update regrets using counterfactual values
                for i in range(2):
                    # Apply appropriate CFR variant formula
                    if Config.CFR_VARIANT == "CFR+":
                        # CFR+ uses positive regrets only
                        self.regrets[infostate][i] = max(0, self.regrets[infostate][i] + 
                                                    counterfactual_reach_prob * (action_utilities[i] - utility))
                    elif Config.CFR_VARIANT == "Linear":
                        # Linear CFR applies linear weighting based on iteration
                        self.regrets[infostate][i] += (self.iteration_count + 1) * counterfactual_reach_prob * \
                                                    (action_utilities[i] - utility)
                    else:
                        # Vanilla CFR
                        self.regrets[infostate][i] += counterfactual_reach_prob * (action_utilities[i] - utility)
                
                # Update strategy sum for average policy computation
                if Config.WEIGHTED_UPDATES:
                    # Accumulate weighted strategy for averaging
                    weight = self.iteration_count + 1
                    self.strategy_sum[infostate] += weight * pr_1 * strategy
                else:
                    # Standard accumulation
                    self.strategy_sum[infostate] += pr_1 * strategy
            
            return utility
            
        else:  # player == 1
            # Similar implementation for player 1 - use the same abstraction approach
            if Config.BUCKETS >= Config.DECK_SIZE:
                bucket = cards[player]  # Jack=0, Queen=1, King=2
            else:
                bucket = cards[player] // (Config.DECK_SIZE // Config.BUCKETS)
            
            infostate = (bucket, tuple(history))
            strategy = self.get_strategy(infostate)
            
            action_utilities = np.zeros(2)
            
            for i, action in enumerate(["check", "bet"]):
                action_history = history + [action]
                new_pr_2 = pr_2 * strategy[i]
                action_utilities[i] = self.compute_cfr(cards, action_history, pr_1, new_pr_2, reach_pr)
            
            utility = np.sum(strategy * action_utilities)
            
            if self.iteration_count < Config.CFR_ITERATIONS:
                counterfactual_reach_prob = pr_1
                
                for i in range(2):
                    if Config.CFR_VARIANT == "CFR+":
                        self.regrets[infostate][i] = max(0, self.regrets[infostate][i] + 
                                                    counterfactual_reach_prob * (action_utilities[i] - utility))
                    elif Config.CFR_VARIANT == "Linear":
                        self.regrets[infostate][i] += (self.iteration_count + 1) * counterfactual_reach_prob * \
                                                    (action_utilities[i] - utility)
                    else:
                        self.regrets[infostate][i] += counterfactual_reach_prob * (action_utilities[i] - utility)
                
                if Config.WEIGHTED_UPDATES:
                    weight = self.iteration_count + 1
                    self.strategy_sum[infostate] += weight * pr_2 * strategy
                else:
                    self.strategy_sum[infostate] += pr_2 * strategy
            
            return utility
    
    def train(self):
        """
        Train the CFR agent through self-play.
        
        This implements the core CFR algorithm:
        1. Start with a uniform random strategy
        2. For many iterations, simulate games against itself
        3. After each game, update regrets and strategy
        4. Return the average strategy which approaches a Nash equilibrium
        """
        print(f"Training CFR agent using {Config.CFR_VARIANT} variant with {Config.BUCKETS} buckets...")
        
        # Track progress with tqdm
        progress_bar = tqdm(range(Config.CFR_ITERATIONS), desc="CFR Training")
        
        for t in progress_bar:
            self.iteration_count = t
            
            # Generate random cards for both players (simulate a new game)
            # Ensure we're using values 0-2 (Jack, Queen, King)
            cards = []
            while len(cards) < 2:
                card = random.randint(0, Config.CARD_VALUES - 1)  # Ensure 0-based for Jack, Queen, King
                if card not in cards:  # Ensure no duplicate cards
                    cards.append(card)
            
            # Run a single CFR iteration (one simulated game)
            # This updates regrets for all decisions in this game
            self.compute_cfr(cards, [], 1.0, 1.0)
            
            # Optionally prune low-regret actions for efficiency
            if Config.USE_PRUNING and t > 0 and t % Config.PRUNING_THRESHOLD == 0:
                self._prune_regrets()
            
            # Update progress bar information
            if t % 5000 == 0:
                avg_strategy = self.get_average_strategy()
                info_count = len(avg_strategy)
                progress_bar.set_postfix({"info_sets": info_count})
                    
        # Final average strategy
        # This is what converges to a Nash equilibrium in two-player zero-sum games
        self.strategy = self.get_average_strategy()
        
        print(f"CFR training complete. Found strategies for {len(self.strategy)} information sets.")
        return self.strategy
    
    def _prune_regrets(self):
        """Prune regrets and strategies with negligible values to save memory"""
        keys_to_prune = []
        for key, regrets in self.regrets.items():
            max_regret = np.max(np.abs(regrets))
            if max_regret < 0.0001:  # Threshold for pruning
                keys_to_prune.append(key)
        
        for key in keys_to_prune:
            del self.regrets[key]
            if key in self.strategy_sum:
                del self.strategy_sum[key]
    
    def get_average_strategy(self):
        """
        Return the average strategy over all iterations.
        
        In CFR, the average strategy converges to a Nash equilibrium, not the 
        current strategy based on regrets. This is a key insight of the algorithm.
        """
        avg_strategy = {}
        
        for infostate, strategy_sum in self.strategy_sum.items():
            total = np.sum(strategy_sum)
            if total > 0:
                avg_strategy[infostate] = strategy_sum / total
            else:
                # Uniform random as fallback
                avg_strategy[infostate] = np.ones(2) / 2
                
        return avg_strategy
    
    def save(self, filename):
        """Save the current strategy to a file"""
        save_strategy(self.get_average_strategy(), filename)
        
    def load(self, filename):
        """Load a strategy from a file"""
        self.strategy = load_strategy(filename)
        return self.strategy
