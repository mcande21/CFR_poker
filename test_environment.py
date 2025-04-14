import unittest
import random
from environment import KuhnPokerEnv

class TestKuhnPokerEnv(unittest.TestCase):
    
    def setUp(self):
        """Initialize the environment"""
        # Set seed for reproducibility
        random.seed(42)
        self.env = KuhnPokerEnv()
        print("\n" + "="*70)
    
    def test_initialization(self):
        """Test that the environment initializes correctly"""
        print("\nTEST: Initialization")
        state = self.env.reset()
        card, history = state
        
        print(f"Player's card: {self.card_name(card)}")
        print(f"Initial pot: {self.env.pot}")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {self.env.terminal}")
        
        # Check initial state
        self.assertIsNotNone(card)
        self.assertIn(card, [0, 1, 2])  # Card should be Jack (0), Queen (1), or King (2)
        self.assertEqual(history, ())  # History should be empty tuple
        self.assertEqual(self.env.pot, 2)  # Pot should start with 2 chips
        self.assertEqual(self.env.current_player, 0)  # First player should be player 0
        self.assertFalse(self.env.terminal)  # Game should not be terminal yet
    
    def test_deck_creation(self):
        """Test that the deck is created properly"""
        print("\nTEST: Deck Creation")
        self.env.reset()
        
        # Check deck composition
        jacks = sum(1 for card in self.env.deck if card == 0)
        queens = sum(1 for card in self.env.deck if card == 1)
        kings = sum(1 for card in self.env.deck if card == 2)
        
        print(f"Deck size: {len(self.env.deck)}")
        print(f"Number of Jacks: {jacks}")
        print(f"Number of Queens: {queens}")
        print(f"Number of Kings: {kings}")
        
        # Each card type should appear 33 times, plus 1 random card
        self.assertEqual(len(self.env.deck), 100)
        
    def test_legal_actions(self):
        """Test that legal actions are returned correctly"""
        print("\nTEST: Legal Actions")
        self.env.reset()
        
        # Initial actions should be check/bet
        actions = self.env.get_legal_actions()
        print(f"Initial legal actions: {actions}")
        self.assertEqual(set(actions), {"check", "bet"})
        
        # After a check, actions should still be check/bet
        self.env.step("check")
        actions = self.env.get_legal_actions()
        print(f"Legal actions after check: {actions}")
        self.assertEqual(set(actions), {"check", "bet"})
        
        # Reset and test after a bet
        self.env.reset()
        self.env.step("bet")
        actions = self.env.get_legal_actions()
        print(f"Legal actions after bet: {actions}")
        self.assertEqual(set(actions), {"call", "fold"})
    
    def test_check_check_sequence(self):
        """Test what happens when both players check"""
        print("\nTEST: Check-Check Sequence")
        self.env.reset()
        
        # Force specific cards for testing
        self.env.hands = [1, 0]  # Player 0 has Queen (1), Player 1 has Jack (0)
        print(f"Player 0 card: {self.card_name(self.env.hands[0])} (Queen)")
        print(f"Player 1 card: {self.card_name(self.env.hands[1])} (Jack)")
        
        # Player 0 checks
        print("\nPlayer 0 checks")
        state, reward, done = self.env.step("check")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.current_player, 1)
        self.assertFalse(done)
        
        # Player 1 checks
        print("\nPlayer 1 checks")
        state, reward, done = self.env.step("check")
        print(f"Terminal state: {done}")
        self.assertTrue(done)  # Game should be terminal after both check
        
        # Player 0 should win with a Queen vs Jack
        print(f"Winner: Player {self.env.winner}")
        print(f"Payoff for Player 0: {self.env.get_payoff(0)}")
        print(f"Payoff for Player 1: {self.env.get_payoff(1)}")
        self.assertEqual(self.env.winner, 0)
        self.assertEqual(self.env.get_payoff(0), 1)
        self.assertEqual(self.env.get_payoff(1), -1)
    
    def test_bet_call_sequence(self):
        """Test what happens when player bets and opponent calls"""
        print("\nTEST: Bet-Call Sequence")
        self.env.reset()
        
        # Force specific cards for testing
        self.env.hands = [0, 2]  # Player 0 has Jack (0), Player 1 has King (2)
        print(f"Player 0 card: {self.card_name(self.env.hands[0])} (Jack)")
        print(f"Player 1 card: {self.card_name(self.env.hands[1])} (King)")
        
        # Player 0 bets
        print("\nPlayer 0 bets")
        state, reward, done = self.env.step("bet")
        print(f"Pot size after bet: {self.env.pot}")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.pot, 3)  # Pot should increase by 1
        self.assertEqual(self.env.current_player, 1)
        self.assertFalse(done)
        
        # Player 1 calls
        print("\nPlayer 1 calls")
        state, reward, done = self.env.step("call")
        print(f"Pot size after call: {self.env.pot}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.pot, 4)  # Pot should increase by 1 more
        self.assertTrue(done)  # Game should be terminal after call
        
        # Player 1 should win with a King vs Jack
        print(f"Winner: Player {self.env.winner}")
        print(f"Payoff for Player 0: {self.env.get_payoff(0)}")
        print(f"Payoff for Player 1: {self.env.get_payoff(1)}")
        self.assertEqual(self.env.winner, 1)
        self.assertEqual(self.env.get_payoff(0), -1)
        self.assertEqual(self.env.get_payoff(1), 1)
    
    def test_bet_fold_sequence(self):
        """Test what happens when player bets and opponent folds"""
        print("\nTEST: Bet-Fold Sequence")
        self.env.reset()
        
        # Force specific cards for testing
        self.env.hands = [2, 0]  # Player 0 has King (2), Player 1 has Jack (0)
        print(f"Player 0 card: {self.card_name(self.env.hands[0])} (King)")
        print(f"Player 1 card: {self.card_name(self.env.hands[1])} (Jack)")
        
        # Player 0 bets
        print("\nPlayer 0 bets")
        state, reward, done = self.env.step("bet")
        print(f"Pot size after bet: {self.env.pot}")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.pot, 3)
        self.assertEqual(self.env.current_player, 1)
        self.assertFalse(done)
        
        # Player 1 folds
        print("\nPlayer 1 folds")
        state, reward, done = self.env.step("fold")
        print(f"Terminal state: {done}")
        self.assertTrue(done)  # Game should be terminal after fold
        
        # Player 0 should win because opponent folded
        print(f"Winner: Player {self.env.winner}")
        print(f"Payoff for Player 0: {self.env.get_payoff(0)}")
        print(f"Payoff for Player 1: {self.env.get_payoff(1)}")
        self.assertEqual(self.env.winner, 0)
        self.assertEqual(self.env.get_payoff(0), 1)
        self.assertEqual(self.env.get_payoff(1), -1)
    
    def test_check_bet_call_sequence(self):
        """Test check-bet-call sequence"""
        print("\nTEST: Check-Bet-Call Sequence")
        self.env.reset()
        
        # Force specific cards for testing
        self.env.hands = [0, 1]  # Player 0 has Jack (0), Player 1 has Queen (1)
        print(f"Player 0 card: {self.card_name(self.env.hands[0])} (Jack)")
        print(f"Player 1 card: {self.card_name(self.env.hands[1])} (Queen)")
        
        # Player 0 checks
        print("\nPlayer 0 checks")
        state, reward, done = self.env.step("check")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.current_player, 1)
        self.assertFalse(done)
        
        # Player 1 bets
        print("\nPlayer 1 bets")
        state, reward, done = self.env.step("bet")
        print(f"Pot size after bet: {self.env.pot}")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.pot, 3)
        self.assertEqual(self.env.current_player, 0)
        self.assertFalse(done)
        
        # Player 0 calls
        print("\nPlayer 0 calls")
        state, reward, done = self.env.step("call")
        print(f"Pot size after call: {self.env.pot}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.pot, 4)
        self.assertTrue(done)
        
        # Player 1 should win with a Queen vs Jack
        print(f"Winner: Player {self.env.winner}")
        print(f"Payoff for Player 0: {self.env.get_payoff(0)}")
        print(f"Payoff for Player 1: {self.env.get_payoff(1)}")
        self.assertEqual(self.env.winner, 1)
        self.assertEqual(self.env.get_payoff(0), -1)
        self.assertEqual(self.env.get_payoff(1), 1)
    
    def test_check_bet_fold_sequence(self):
        """Test check-bet-fold sequence"""
        print("\nTEST: Check-Bet-Fold Sequence")
        self.env.reset()
        
        # Force specific cards for testing
        self.env.hands = [0, 1]  # Player 0 has Jack (0), Player 1 has Queen (1)
        print(f"Player 0 card: {self.card_name(self.env.hands[0])} (Jack)")
        print(f"Player 1 card: {self.card_name(self.env.hands[1])} (Queen)")
        
        # Player 0 checks
        print("\nPlayer 0 checks")
        state, reward, done = self.env.step("check")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.current_player, 1)
        self.assertFalse(done)
        
        # Player 1 bets
        print("\nPlayer 1 bets")
        state, reward, done = self.env.step("bet")
        print(f"Pot size after bet: {self.env.pot}")
        print(f"Current player: {self.env.current_player}")
        print(f"Terminal state: {done}")
        self.assertEqual(self.env.pot, 3)
        self.assertEqual(self.env.current_player, 0)
        self.assertFalse(done)
        
        # Player 0 folds
        print("\nPlayer 0 folds")
        state, reward, done = self.env.step("fold")
        print(f"Terminal state: {done}")
        self.assertTrue(done)
        
        # Player 1 should win because opponent folded
        print(f"Winner: Player {self.env.winner}")
        print(f"Payoff for Player 0: {self.env.get_payoff(0)}")
        print(f"Payoff for Player 1: {self.env.get_payoff(1)}")
        self.assertEqual(self.env.winner, 1)
        self.assertEqual(self.env.get_payoff(0), -1)
        self.assertEqual(self.env.get_payoff(1), 1)
    
    def test_tie_game(self):
        """Test what happens when players have the same card"""
        print("\nTEST: Tie Game")
        self.env.reset()
        
        # Force same cards for both players
        self.env.hands = [1, 1]  # Both players have Queen (1)
        print(f"Player 0 card: {self.card_name(self.env.hands[0])} (Queen)")
        print(f"Player 1 card: {self.card_name(self.env.hands[1])} (Queen)")
        
        # Player 0 checks
        print("\nPlayer 0 checks")
        state, reward, done = self.env.step("check")
        print(f"Current player: {self.env.current_player}")
        
        # Player 1 checks
        print("\nPlayer 1 checks")
        state, reward, done = self.env.step("check")
        print(f"Terminal state: {done}")
        self.assertTrue(done)
        
        # Should be a tie
        if self.env.hands[0] == self.env.hands[1]:
            print("Cards are tied!")
            print(f"Winner according to implementation: {self.env.winner}")
            print(f"Payoff for Player 0: {self.env.get_payoff(0)}")
            print(f"Payoff for Player 1: {self.env.get_payoff(1)}")
            # In case of tie, neither player should win
            self.assertIsNone(self.env.winner)
            self.assertEqual(self.env.get_payoff(0), 0)
            self.assertEqual(self.env.get_payoff(1), 0)
    
    def card_name(self, card_value):
        """Helper method to convert card value to readable name"""
        if card_value == 0:
            return "Jack"
        elif card_value == 1:
            return "Queen"
        elif card_value == 2:
            return "King"
        else:
            return f"Unknown ({card_value})"

if __name__ == '__main__':
    unittest.main()