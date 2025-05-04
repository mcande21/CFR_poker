# Evnironment for Kuhn Poker
import random

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

    def state(self):
        """ Returns state (card, action_history) for current player """
        card = self.hands[self.current_player]
        return (card, tuple(self.history))

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
            return 0
            
        if player == self.winner:
            return 1
        else:
            return -1
            
    def clone(self):
        """ Creates a deep copy of the current environment state """
        clone_env = KuhnPokerEnv()
        clone_env.deck = self.deck.copy()
        clone_env.hands = self.hands.copy()
        clone_env.pot = self.pot
        clone_env.current_player = self.current_player
        clone_env.history = self.history.copy()
        clone_env.terminal = self.terminal
        if hasattr(self, 'winner'):
            clone_env.winner = self.winner
        return clone_env
        
    def set_state(self, card_or_env, history=None):
        """ 
        Restores this environment to a specified state
        
        This method can be called in two ways:
        1. set_state(saved_env): Restores from a saved environment
        2. set_state(card, history): Sets the state with specified card and history
        
        Args:
            card_or_env: Either a card value (0=Jack, 1=Queen, 2=King) or a saved environment
            history: Action history when using the card+history form. Not used when restoring from env.
        """
        if history is None:
            # First form: restoring from saved environment
            saved_env = card_or_env
            self.deck = saved_env.deck.copy()
            self.hands = saved_env.hands.copy()
            self.pot = saved_env.pot
            self.current_player = saved_env.current_player
            self.history = saved_env.history.copy()
            self.terminal = saved_env.terminal
            if hasattr(saved_env, 'winner'):
                self.winner = saved_env.winner
        else:
            # Second form: setting up with card and history
            card = card_or_env
            # Create a basic state with the specified card
            self.hands = [card, 0]  # Set opponent card to 0 (Jack) by default
            self.pot = 2  # Default pot
            self.current_player = 0
            
            # Process history to update game state
            self.history = list(history) if isinstance(history, tuple) else list(history)
            self.terminal = False
            
            # Process each action in the history to update pot and current player
            for action in self.history:
                if action == "bet":
                    self.pot += 1
                if action == "call":
                    self.pot += 1
                    self.terminal = True
                    # determine_win would be called here, but we don't have opponent's card
                if action == "fold":
                    self.terminal = True
                    self.winner = 1 - self.current_player
                
                # After each action, switch player except for terminal actions
                if not self.terminal:
                    self.current_player = 1 - self.current_player
            
            # Special case: both players checked
            if len(self.history) == 2 and self.history[0] == "check" and self.history[1] == "check":
                self.terminal = True
                # Winner would be determined by cards, but we don't have opponent's card
        
        return self.state()