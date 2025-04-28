import random
from config import Config
import copy

class KuhnPokerEnv:
    def __init__(self):
        self.deck = []
        self.players = 2
        self.reset()

    def reset(self):
        """ 
        Reset the game:
        - create new deck
        - shuffle the deck
        - deal hands (1 card)
        - init pot (2 chips)
        - set current player
        - init history (empty)
        - set terminal false
        """
        # Create a 100 card deck (33/J, 33/Q, 33/K, + 1 random)
        self.deck = ([0] * 33) + ([1] * 33) + ([2] * 33)
        # Add one extra random card
        self.deck.append(random.choice([0, 1, 2]))
        random.shuffle(self.deck)
        self.hands = [self.deck[0], self.deck[1]] # hands are just first two from deck
        self.pot = 2
        self.current_player = 0
        self.history = []
        self.terminal = False
        return self.state()
        
    def set_cards(self, cards):
        """Set specific cards for the players"""
        if len(cards) == self.players:
            self.hands = cards.copy()
        return self.state()

    def state(self):
        """Returns (bucket_id, action_history) for current player"""
        card = self.hands[self.current_player]
        
        if Config.BUCKETS >= Config.DECK_SIZE:
            # If we have more buckets than cards, each card gets its own bucket
            bucket = card
        else:
            # Otherwise, distribute cards across buckets
            bucket = card // (Config.DECK_SIZE // Config.BUCKETS)
            
        return (bucket, tuple(self.history))

    def get_legal_actions(self):
        """ 
        Return valid actions
        - if there is no history --> check/bet
        - if a player just checked --> check/bet
        - if a player just bet --> call/fold
        """
        if not self.history or (len(self.history) == 1 and self.history[0] == "check"):
            return ["check", "bet"]
        else:
            return ["call", "fold"]

    def step(self, action):
        """ Conduct action """
        self.history.append(action)
        
        # Bets can only be 1 chip
        if action == "bet":
            self.pot += 1
            # switch player
            self.current_player = 1 - self.current_player

        elif action == "check":
            # switch player
            self.current_player = 1 - self.current_player
            # both players just checked --> determine winner
            if len(self.history) == 2:
                self.terminal = True
                self.determine_win()
        # both players just bet --> detemrine winner
        elif action == "call":
            self.pot += 1
            self.terminal = True
            self.determine_win()
        # current player folded --> set winner
        elif action == "fold":
            self.terminal = True
            self.winner = 1 - self.current_player
        
        # return state, reward placeholder, end of game
        return self.state(), 0, self.terminal

    def determine_win(self):
        """ determines the winner """
        if self.hands[0] == self.hands[1]:
            # It's a tie
            self.winner = None
        else:
            self.winner = 0 if self.hands[0] > self.hands[1] else 1

    def get_payoff(self, player):
        """ Calculates reward for agent """
        # game is still going, no reward
        if not self.terminal:
            return 0
            
        # Handle tie case
        if self.winner is None:
            return 0.1
            
        if player == self.winner:
            return 0.25
        else:
            return -0.75

    def clone(self):
        """Create a deep copy of the environment for simulation"""
        new_env = KuhnPokerEnv()
        new_env.deck = self.deck.copy()
        new_env.hands = self.hands.copy()
        new_env.pot = self.pot
        new_env.current_player = self.current_player
        new_env.history = self.history.copy()
        new_env.terminal = self.terminal
        if hasattr(self, 'winner'):
            new_env.winner = self.winner
        return new_env
    
    def set_state(self, bucket, history):
        """Set the environment to a specific state for counterfactual simulations"""
        # Reset first
        self.pot = 2  # Starting pot (both players ante 1)
        self.current_player = 0
        self.history = []
        self.terminal = False
        
        # Set cards based on bucket
        # In Kuhn poker, bucket is the card value (0=Jack, 1=Queen, 2=King)
        opponent_cards = [i for i in range(3) if i != bucket]
        opponent_card = random.choice(opponent_cards)
        self.hands = [bucket, opponent_card]
        
        # Replay the history actions to reach the desired state
        current_history = list(history)
        for action in current_history:
            self.step(action)
    
    def is_terminal(self):
        """Check if the game is in a terminal state"""
        return self.terminal