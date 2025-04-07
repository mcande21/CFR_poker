import random
from config import Config

class KuhnPokerEnv:
    def __init__(self):
        self.deck = []
        self.players = 2
        self.reset()

    def reset(self):
        # Create the deck with 33 cards of each value (0, 1, 2) plus 1 random card
        self.deck = ([0] * 33) + ([1] * 33) + ([2] * 33)
        # Add one extra random card
        self.deck.append(random.choice([0, 1, 2]))
        random.shuffle(self.deck)
        self.hands = [self.deck[0], self.deck[1]]
        self.pot = 2  # Ante of 1 from each player
        self.current_player = 0
        self.history = []
        self.terminal = False
        return self.infostate()
        
    def set_cards(self, cards):
        """Set specific cards for the players (used during training)"""
        if len(cards) == self.players:
            self.hands = cards.copy()
        return self.infostate()

    def infostate(self):
        """Returns (bucket_id, action_history) for current player"""
        card = self.hands[self.current_player]
        
        # Fix the division by zero error by handling the case when BUCKETS >= DECK_SIZE
        if Config.BUCKETS >= Config.DECK_SIZE:
            # If we have more buckets than cards, each card gets its own bucket
            bucket = card
        else:
            # Otherwise, distribute cards across buckets
            bucket = card // (Config.DECK_SIZE // Config.BUCKETS)
            
        return (bucket, tuple(self.history))

    def get_legal_actions(self):
        return ["check", "bet"] if not self.history else ["call", "fold"]

    def step(self, action):
        self.history.append(action)
        
        if action == "bet":
            self.pot += 1
            self.current_player = 1 - self.current_player
        elif action == "check":
            self.current_player = 1 - self.current_player
            if len(self.history) == 2:  # Both players checked
                self.terminal = True
                self._resolve_showdown()
        elif action == "call":
            self.pot += 1
            self.terminal = True
            self._resolve_showdown()
        elif action == "fold":
            self.terminal = True
            self.winner = 1 - self.current_player
            
        return self.infostate(), 0, self.terminal

    def _resolve_showdown(self):
        self.winner = 0 if self.hands[0] > self.hands[1] else 1

    def get_payoff(self, player):
        if not self.terminal:
            return 0
            
        if "fold" in self.history:
            return self.pot // 2 if player == self.winner else -self.pot // 2
        
        # Showdown
        if player == self.winner:
            return self.pot // 2
        else:
            return -self.pot // 2