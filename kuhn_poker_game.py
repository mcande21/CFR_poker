import pygame
import sys
import random
import json
import os
import numpy as np
from config import Config
from royal_poker.utils import load_strategy
from debug_strategy import print_strategy_breakdown

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
CARD_WIDTH, CARD_HEIGHT = 120, 170
BUTTON_WIDTH, BUTTON_HEIGHT = 120, 50

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (34, 139, 34)
DARK_GREEN = (0, 100, 0)
RED = (220, 20, 20)
BLUE = (30, 70, 190)
LIGHT_BLUE = (70, 130, 180)
GOLD = (218, 165, 32)
CREAM = (255, 253, 208)

# Card suits and their colors
SUITS = ['♠', '♥', '♦', '♣']
SUIT_COLORS = {
    '♠': BLACK,
    '♥': RED,
    '♦': RED,
    '♣': BLACK
}

class KuhnPokerGame:
    def __init__(self, strategy):
        self.strategy = strategy
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Kuhn Poker")
        
        # Load fonts
        self.font = pygame.font.SysFont('Arial', 24)
        self.big_font = pygame.font.SysFont('Arial', 36)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # Game state
        self.player_chips = 10
        self.ai_chips = 10
        self.reset_game()
        
        # Create card visuals
        self.card_images = self.create_card_images()
        self.card_back = self.create_card_back()
        
        # Create UI elements
        self.buttons = {
            'bet': pygame.Rect(WIDTH//2 - 130, HEIGHT - 80, BUTTON_WIDTH, BUTTON_HEIGHT),
            'check': pygame.Rect(WIDTH//2 + 10, HEIGHT - 80, BUTTON_WIDTH, BUTTON_HEIGHT),
            'call': pygame.Rect(WIDTH//2 - 130, HEIGHT - 80, BUTTON_WIDTH, BUTTON_HEIGHT),
            'fold': pygame.Rect(WIDTH//2 + 10, HEIGHT - 80, BUTTON_WIDTH, BUTTON_HEIGHT),
            'next_round': pygame.Rect(WIDTH//2 - 80, HEIGHT - 150, 160, BUTTON_HEIGHT),
            'new_game': pygame.Rect(WIDTH//2 - 80, HEIGHT - 80, 160, BUTTON_HEIGHT),
            'debug': pygame.Rect(WIDTH - 100, 10, 80, 30)  # Debug button
        }
        
        # Animation state
        self.animation_active = False
        self.animation_progress = 0
        self.show_result = False
        
        # Debug information
        self.debug_mode = False
        self.last_strategy_info = None
        
    def reset_game(self):
        self.player_chips = 10
        self.ai_chips = 10
        self.start_new_round()
    
    def start_new_round(self):
        # Create deck with the same distribution as the training environment (33 of each card + 1 random)
        self.deck = ([0] * 33) + ([1] * 33) + ([2] * 33)
        # Add one extra random card to match env.py
        self.deck.append(random.choice([0, 1, 2]))

        random.shuffle(self.deck)
        
        # Deal cards
        self.player_card = self.deck[0]
        self.ai_card = self.deck[1]
        
        # Game state
        self.pot = 2  # Ante of 1 from each player
        self.player_chips -= 1
        self.ai_chips -= 1
        
        self.history = []
        self.game_over = False
        self.winner = None
        self.player_turn = True
        self.message = "Your turn. Check or Bet?"
        self.round_stage = "first_action"
        
        # Reset animation
        self.animation_active = True
        self.animation_progress = 0
        self.show_result = False
        
    def create_card_images(self):
        """Create card images for all cards in the deck"""
        card_images = {}
        for card in range(Config.DECK_SIZE):
            value, suit = self.get_card_name(card)
            card_surf = self.create_card(value, suit)
            card_images[card] = card_surf
        return card_images
    
    def create_card(self, value, suit):
        """Create a single card image"""
        card_surf = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
        card_surf.fill(CREAM)
        
        # Draw card border
        pygame.draw.rect(card_surf, BLACK, (0, 0, CARD_WIDTH, CARD_HEIGHT), 2)
        
        # Draw card value and suit
        value_text = self.big_font.render(value, True, SUIT_COLORS[suit])
        suit_text = self.big_font.render(suit, True, SUIT_COLORS[suit])
        
        # Center of card
        card_surf.blit(value_text, (CARD_WIDTH//2 - value_text.get_width()//2, 
                                   CARD_HEIGHT//2 - value_text.get_height()//2 - 15))
        card_surf.blit(suit_text, (CARD_WIDTH//2 - suit_text.get_width()//2, 
                                 CARD_HEIGHT//2 + 5))
        
        # Corner indicators
        small_value = self.small_font.render(value, True, SUIT_COLORS[suit])
        small_suit = self.small_font.render(suit, True, SUIT_COLORS[suit])
        
        # Top-left
        card_surf.blit(small_value, (5, 5))
        card_surf.blit(small_suit, (5, 25))
        
        # Bottom-right (rotated)
        rotated_value = pygame.transform.rotate(small_value, 180)
        rotated_suit = pygame.transform.rotate(small_suit, 180)
        card_surf.blit(rotated_value, (CARD_WIDTH - rotated_value.get_width() - 5, 
                                     CARD_HEIGHT - rotated_value.get_height() - 25))
        card_surf.blit(rotated_suit, (CARD_WIDTH - rotated_suit.get_width() - 5, 
                                     CARD_HEIGHT - rotated_suit.get_height() - 5))
        
        return card_surf
    
    def create_card_back(self):
        """Create card back design"""
        card_back = pygame.Surface((CARD_WIDTH, CARD_HEIGHT))
        card_back.fill(RED)
        
        # Draw card border
        pygame.draw.rect(card_back, BLACK, (0, 0, CARD_WIDTH, CARD_HEIGHT), 2)
        
        # Draw pattern on back
        for i in range(10, CARD_WIDTH-10, 10):
            pygame.draw.line(card_back, (128, 0, 0), 
                            (i, 10), (i, CARD_HEIGHT-10), 2)
        
        return card_back
    
    def get_card_name(self, card):
        """Convert card value to readable name and suit"""
        # Card values are 0=Jack, 1=Queen, 2=King
        value = card % Config.CARD_VALUES
        
        # Determine suit based on position in deck
        suit_idx = card // Config.CARD_VALUES % len(SUITS)
        suit = SUITS[suit_idx]
        
        # Convert numerical value to card name
        if value == 0:
            return "J", suit
        elif value == 1:
            return "Q", suit
        else:
            return "K", suit
    
    def player_action(self, action):
        """Process player's action"""
        if self.game_over or not self.player_turn:
            return
        
        self.history.append(action)
        
        if action == "bet":
            self.player_chips -= 1
            self.pot += 1
            self.message = "You bet 1 chip."
            
        elif action == "check":
            self.message = "You checked."
            
        # Process round stage transitions
        if self.round_stage == "first_action":
            self.round_stage = "second_action"
        elif self.round_stage == "second_action" and action == "check" and len(self.history) >= 2 and self.history[-2] == "check":
            # Both players checked, go to showdown
            self.message = "Both players checked. Going to showdown."
            self.determine_winner()
            return
        elif self.round_stage == "second_action" and action == "bet":
            self.round_stage = "response"
        
        # Now it's AI's turn
        self.player_turn = False
        self.ai_move()
    
    def ai_move(self):
        """AI makes a decision based on strategy"""
        # Determine AI's bucket
        if Config.BUCKETS >= Config.DECK_SIZE:
            bucket = self.ai_card
        else:
            bucket = self.ai_card // (Config.DECK_SIZE // Config.BUCKETS)

        
        history_tuple = tuple(self.history)
        info_set_key = f"({bucket}, {history_tuple})"
        
        # Get AI's action from strategy
        action, strategy_info = self.get_ai_action(info_set_key, bucket, return_info=True)
        self.last_strategy_info = strategy_info
        self.history.append(action)
        
        # Process AI's action
        if self.round_stage == "second_action" and len(self.history) >= 2 and self.history[-2] == "bet":
            # AI is responding to a player's bet
            if action == "bet":  # AI calls
                self.ai_chips -= 1
                self.pot += 1
                self.message = "AI calls your bet. Going to showdown."
                self.determine_winner()
                return
            else:  # AI folds
                self.winner = "Player"
                self.player_chips += self.pot
                self.pot = 0
                self.game_over = True
                self.message = "AI folds. You win the pot!"
                self.show_result = True
                return
        else:
            # Standard bet/check action
            if action == "bet":
                self.ai_chips -= 1
                self.pot += 1
                self.message = "AI bets 1 chip."
            else:
                self.message = "AI checks."
        
        # Process round progression
        if self.round_stage == "first_action":
            self.round_stage = "second_action"
            if action == "bet":
                self.message = "AI bets. Call or Fold?"
                self.player_turn = True
            else:
                self.message = "AI checks. Check or Bet?"
                self.player_turn = True
                
        elif self.round_stage == "second_action":
            if action == "bet":
                # If player checked and AI bets, player gets to respond
                self.message = "AI bets. Call or Fold?"
                self.round_stage = "response"
                self.player_turn = True
            else:  # action == "check"
                if len(self.history) >= 2 and self.history[-2] == "check":
                    # Both checked, go to showdown
                    self.message = "Both players checked. Going to showdown."
                    self.determine_winner()
                else:
                    self.player_turn = True
    
    def get_ai_action(self, info_set_key, bucket, return_info=False):
        """Determine AI's action based on strategy"""
        strategy_info = {
            "key": info_set_key,
            "bucket": bucket,
            "probabilities": None,
            "source": "fallback",
            "raw_key": None
        }
        
        # Try to find exact strategy for this state
        if info_set_key in self.strategy:
            probs = self.strategy[info_set_key]
            strategy_info["probabilities"] = probs
            strategy_info["source"] = "exact match"
            strategy_info["raw_key"] = info_set_key
            
            # Check if we're responding to a bet
            is_response_to_bet = len(self.history) > 0 and self.history[-1] == "bet"
            
            # Select action based on probabilities
            action = random.choices(["check", "bet"], weights=probs, k=1)[0]
            
            # Translate abstract action to concrete action if necessary
            if is_response_to_bet:
                # When responding to a bet, "check" means "fold" and "bet" means "call"
                # Log to help debugging
                action_meaning = "fold" if action == "check" else "call"
                print(f"AI responding to bet with {action} (meaning {action_meaning}), probs={probs}")
            else:
                print(f"AI initiating with {action}, probs={probs}")
            
            if return_info:
                return action, strategy_info
            return action
        
        # Try simplified key (just based on last action)
        if len(self.history) > 0:
            simplified_key = f"({bucket}, ('{self.history[-1]}',))"
            if simplified_key in self.strategy:
                probs = self.strategy[simplified_key]
                strategy_info["probabilities"] = probs
                strategy_info["source"] = "simplified key"
                strategy_info["raw_key"] = simplified_key
                action = random.choices(["check", "bet"], weights=probs, k=1)[0]
                
                if return_info:
                    return action, strategy_info
                return action
        
        # If no strategy found, use basic heuristic
        if len(self.history) > 0 and self.history[-1] == "bet":
            # Responding to a bet - interpret as call/fold decision
            if self.ai_card > 0:  # Queens and Kings call
                action = "bet"  # representing "call" in this context
                prob = 0.9 if self.ai_card == 2 else 0.7
                strategy_info["probabilities"] = [1.0 - prob, prob]
            else:
                prob = 0.3  # Usually fold with Jacks
                action = "check" if random.random() < 0.7 else "bet"
                strategy_info["probabilities"] = [0.7, 0.3]
        else:
            # No bet to respond to
            if self.ai_card == 2:  # King
                prob = 0.7
                action = "bet" if random.random() < prob else "check"
                strategy_info["probabilities"] = [1.0 - prob, prob]
            elif self.ai_card == 1:  # Queen
                prob = 0.3
                action = "bet" if random.random() < prob else "check"
                strategy_info["probabilities"] = [1.0 - prob, prob]
            else:  # Jack
                prob = 0.1
                action = "bet" if random.random() < prob else "check"
                strategy_info["probabilities"] = [1.0 - prob, prob]
        
        strategy_info["source"] = "heuristic"
        
        if return_info:
            return action, strategy_info
        return action
    
    def call(self):
        """Player calls AI's bet"""
        if self.game_over or not self.player_turn:
            return
            
        if not (len(self.history) > 0 and self.history[-1] == "bet"):
            return
            
        self.player_chips -= 1
        self.pot += 1
        self.history.append("call")
        self.message = "You call AI's bet. Going to showdown."
        self.determine_winner()
    
    def fold(self):
        """Player folds"""
        if self.game_over or not self.player_turn:
            return
            
        if not (len(self.history) > 0 and self.history[-1] == "bet"):
            return
            
        self.history.append("fold")
        self.winner = "AI"
        self.ai_chips += self.pot
        self.pot = 0
        self.game_over = True
        self.message = "You folded. AI wins the pot!"
        self.show_result = True
    
    def determine_winner(self):
        """Determine winner at showdown"""
        self.game_over = True
        self.show_result = True
        
        if self.player_card > self.ai_card:
            self.winner = "Player"
            self.player_chips += self.pot
            self.message = f"You win with {self.get_card_name(self.player_card)[0]}!"
        elif self.player_card < self.ai_card:
            self.winner = "AI"
            self.ai_chips += self.pot
            self.message = f"AI wins with {self.get_card_name(self.ai_card)[0]}!"
        else:
            # It's a tie
            self.winner = "Tie"
            # Split the pot
            split = self.pot // 2
            self.player_chips += split
            self.ai_chips += split
            # Handle odd chip
            if self.pot % 2 == 1:
                if random.choice([True, False]):
                    self.player_chips += 1
                    self.message = f"Tie! Pot split (you got the extra chip)."
                else:
                    self.ai_chips += 1
                    self.message = f"Tie! Pot split (AI got the extra chip)."
            else:
                self.message = f"Tie! Pot split evenly."
                
        self.pot = 0
    
    def draw_button(self, button_key, text, hover=False, disabled=False):
        """Draw a button with the given text"""
        button_rect = self.buttons[button_key]
        
        if disabled:
            color = (150, 150, 150)  # Gray for disabled
        elif hover:
            color = LIGHT_BLUE
        else:
            color = BLUE
            
        pygame.draw.rect(self.screen, color, button_rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, button_rect, 2, border_radius=10)  # Border
        
        font = self.small_font if button_key == 'debug' else self.font
        text_surf = font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=button_rect.center)
        self.screen.blit(text_surf, text_rect)
    
    def draw_card(self, card, x, y, is_back=False):
        """Draw a card at the specified position"""
        if is_back:
            self.screen.blit(self.card_back, (x, y))
        else:
            self.screen.blit(self.card_images[card], (x, y))
    
    def draw_chips(self, amount, x, y):
        """Draw chips representing an amount"""
        if amount <= 0:
            return
            
        chip_radius = 15
        chip_colors = [(255, 0, 0), (0, 0, 255)]  # Red for player, blue for AI
        
        for i in range(amount):
            # Stagger the chips
            chip_x = x + (i % 4) * 5
            chip_y = y - (i // 4) * 5
            
            # Alternate colors
            color = chip_colors[i % len(chip_colors)]
            
            # Draw chip
            pygame.draw.circle(self.screen, color, (chip_x, chip_y), chip_radius)
            pygame.draw.circle(self.screen, (color[0]//2, color[1]//2, color[2]//2), 
                              (chip_x, chip_y), chip_radius-2, 2)
    
    def draw_debug_info(self):
        """Draw debug information about AI's strategy"""
        if not self.debug_mode or not self.last_strategy_info:
            return
            
        # Create a debug panel
        panel_rect = pygame.Rect(WIDTH - 270, 50, 250, 260)
        pygame.draw.rect(self.screen, (50, 50, 50, 200), panel_rect)
        pygame.draw.rect(self.screen, BLACK, panel_rect, 2)
        
        # Title
        title = self.small_font.render("AI Strategy Debug", True, WHITE)
        self.screen.blit(title, (panel_rect.x + 10, panel_rect.y + 10))
        
        # AI card info
        if Config.BUCKETS >= Config.DECK_SIZE:
            bucket = self.ai_card
        else:
            bucket = self.ai_card // (Config.DECK_SIZE // Config.BUCKETS)
        
        card_name = self.get_card_name(self.ai_card)[0] if self.game_over else "?"
        card_info = self.small_font.render(f"AI Card: {card_name} (Bucket: {bucket})", True, WHITE)
        self.screen.blit(card_info, (panel_rect.x + 10, panel_rect.y + 40))
        
        # History
        history_text = self.small_font.render(f"History: {tuple(self.history)}", True, WHITE)
        self.screen.blit(history_text, (panel_rect.x + 10, panel_rect.y + 70))
        
        # Strategy source
        source = self.last_strategy_info["source"]
        source_text = self.small_font.render(f"Strategy: {source}", True, GOLD)
        self.screen.blit(source_text, (panel_rect.x + 10, panel_rect.y + 100))
        
        if self.last_strategy_info["raw_key"]:
            key_text = self.small_font.render(f"Key: {self.last_strategy_info['raw_key'][:20]}...", True, WHITE)
            self.screen.blit(key_text, (panel_rect.x + 10, panel_rect.y + 130))
        
        # Probabilities
        probs = self.last_strategy_info["probabilities"]
        if probs:
            # Draw probability bars
            check_text = self.small_font.render(f"Check: {probs[0]:.2f}", True, WHITE)
            bet_text = self.small_font.render(f"Bet: {probs[1]:.2f}", True, WHITE)
            
            self.screen.blit(check_text, (panel_rect.x + 10, panel_rect.y + 160))
            self.screen.blit(bet_text, (panel_rect.x + 10, panel_rect.y + 190))
            
            # Visualize probabilities
            bar_width = 150
            bar_height = 15
            bar_x = panel_rect.x + 80
            
            # Check probability bar
            pygame.draw.rect(self.screen, (100, 100, 100), 
                            (bar_x, panel_rect.y + 165, bar_width, bar_height))
            pygame.draw.rect(self.screen, (0, 200, 0), 
                            (bar_x, panel_rect.y + 165, int(bar_width * probs[0]), bar_height))
            
            # Bet probability bar
            pygame.draw.rect(self.screen, (100, 100, 100), 
                            (bar_x, panel_rect.y + 195, bar_width, bar_height))
            pygame.draw.rect(self.screen, (200, 0, 0), 
                            (bar_x, panel_rect.y + 195, int(bar_width * probs[1]), bar_height))
    
    def run(self):
        """Main game loop"""
        clock = pygame.time.Clock()
        running = True
        
        # For button hover effects
        hover_states = {key: False for key in self.buttons}
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # Mouse position for hover effects
                mouse_pos = pygame.mouse.get_pos()
                
                # Update hover states
                for key, rect in self.buttons.items():
                    hover_states[key] = rect.collidepoint(mouse_pos)
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    # Debug button toggle
                    if hover_states['debug']:
                        self.debug_mode = not self.debug_mode
                        print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
                        
                        # Print strategy info if turning on debug
                        if self.debug_mode and len(self.strategy) > 0:
                            print(f"Strategy has {len(self.strategy)} entries")
                            
                    # Game is over, handle restart buttons
                    if self.game_over:
                        if hover_states['new_game']:
                            self.reset_game()
                        elif hover_states['next_round'] and self.player_chips > 0 and self.ai_chips > 0:
                            self.start_new_round()
                    
                    # Game in progress, handle action buttons
                    elif self.player_turn:
                        if self.round_stage == "response" or (len(self.history) > 0 and self.history[-1] == "bet"):
                            # Call or fold
                            if hover_states['call']:
                                self.call()
                            elif hover_states['fold']:
                                self.fold()
                        else:
                            # Check or bet
                            if hover_states['check']:
                                self.player_action("check")
                            elif hover_states['bet']:
                                self.player_action("bet")
            
            # Update animation
            if self.animation_active:
                self.animation_progress += 0.1
                if self.animation_progress >= 1:
                    self.animation_active = False
                    self.animation_progress = 1
            
            # Draw the game
            self.draw_game(hover_states)
            
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()
    
    def draw_game(self, hover_states):
        """Draw the game state"""
        # Clear screen
        self.screen.fill(GREEN)
        
        # Draw table
        pygame.draw.ellipse(self.screen, DARK_GREEN, (50, 50, WIDTH-100, HEIGHT-100))
        pygame.draw.ellipse(self.screen, BLACK, (50, 50, WIDTH-100, HEIGHT-100), 2)
        
        # Draw cards
        card_x = WIDTH//2 - CARD_WIDTH//2
        
        # Player card
        if self.animation_active:
            # Card dealing animation
            progress = self.animation_progress
            self.draw_card(self.player_card, 
                         card_x - (1-progress) * 200, 
                         HEIGHT - 220)
        else:
            self.draw_card(self.player_card, card_x, HEIGHT - 220)
        
        # AI card
        if self.animation_active:
            self.draw_card(self.ai_card if self.game_over else None, 
                         card_x - (1-self.animation_progress) * 200, 
                         70, 
                         is_back=not self.game_over)
        else:
            self.draw_card(self.ai_card if self.game_over else None, 
                         card_x, 70, 
                         is_back=not self.game_over)
        
        # Draw chips
        if self.pot > 0:
            self.draw_chips(self.pot, WIDTH//2, HEIGHT//2)
        
        # Draw player stats
        pygame.draw.rect(self.screen, DARK_GREEN, (20, HEIGHT-50, 150, 40), border_radius=5)
        chips_text = self.font.render(f"Your chips: {self.player_chips}", True, WHITE)
        self.screen.blit(chips_text, (30, HEIGHT-45))
        
        pygame.draw.rect(self.screen, DARK_GREEN, (20, 10, 150, 40), border_radius=5)
        ai_text = self.font.render(f"AI chips: {self.ai_chips}", True, WHITE)
        self.screen.blit(ai_text, (30, 15))
        
        # Draw pot
        pot_text = self.font.render(f"Pot: {self.pot}", True, WHITE)
        pot_rect = pot_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 40))
        self.screen.blit(pot_text, pot_rect)
        
        # Draw message
        msg_bg = pygame.Rect(WIDTH//2 - 250, HEIGHT//2 + 50, 500, 40)
        pygame.draw.rect(self.screen, DARK_GREEN, msg_bg, border_radius=10)
        msg_text = self.font.render(self.message, True, WHITE)
        msg_rect = msg_text.get_rect(center=msg_bg.center)
        self.screen.blit(msg_text, msg_rect)
        
        # Draw buttons based on game state
        if self.game_over:
            self.draw_button('new_game', "New Game", hover_states['new_game'])
            if self.player_chips > 0 and self.ai_chips > 0:
                self.draw_button('next_round', "Next Round", hover_states['next_round'])
        elif self.player_turn:
            if self.round_stage == "response" or (len(self.history) > 0 and self.history[-1] == "bet"):
                # Call or fold buttons
                self.draw_button('call', "Call", hover_states['call'])
                self.draw_button('fold', "Fold", hover_states['fold'])
            else:
                # Check or bet buttons
                self.draw_button('check', "Check", hover_states['check'])
                self.draw_button('bet', "Bet", hover_states['bet'])
        
        # Draw debug button
        self.draw_button('debug', "Debug", hover_states['debug'], disabled=False)
        
        # Draw debug panel if enabled
        if self.debug_mode:
            self.draw_debug_info()

def load_best_strategy():
    """Load the best available strategy"""
    strategy_files = [
        os.path.join("strategy", "3B_CFR+_strategy.json"),  # Best choice with correct abstraction
        os.path.join("strategy", "3B_Vanilla_strategy.json"),  # Second choice
        os.path.join("strategy", "3B_Linear_strategy.json")   # Third choice
    ]
    
    for path in strategy_files:
        if os.path.exists(path):
            try:
                print(f"Loading strategy from: {path}")
                with open(path, 'r') as f:
                    strategy = json.load(f)
                    # Convert string keys to proper format for lookups
                    processed_strategy = {}
                    for key, value in strategy.items():
                        # The keys are stored as strings but need to be converted back to the proper format
                        # e.g., "(0, ('check',))" -> (0, ('check',))
                        try:
                            # Extract the bucket number and history tuple from the string
                            parts = key.strip("()").split(", ", 1)
                            bucket = int(parts[0])
                            history_str = parts[1] if len(parts) > 1 else "()"
                            # Convert history string to actual tuple
                            history = eval(history_str)
                            processed_strategy[(bucket, history)] = value
                        except:
                            # Fall back to using string keys if parsing fails
                            processed_strategy[key] = value
                    
                    print(f"Loaded strategy with {len(processed_strategy)} states")
                    return processed_strategy
            except Exception as e:
                print(f"Error loading strategy: {e}")
                continue
    
    # Fallback to empty strategy if none found
    print("No strategy files found. Using basic strategy.")
    return {}

def main():
    strategy = load_best_strategy()
    game = KuhnPokerGame(strategy)
    game.run()

if __name__ == "__main__":
    main()
