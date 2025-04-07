# Extended Kuhn Poker with 100-Card Deck

This project implements an extended version of Kuhn poker using a 100-card deck with repeating Jack, Queen, King values. This approach significantly increases the game's complexity and state space while maintaining the familiar Kuhn poker structure.

## Game Rules

Extended Kuhn Poker follows the same basic rules as standard Kuhn Poker:
- Each player is dealt one card from the deck
- Players ante 1 chip to start
- First player can check or bet (betting costs 1 chip)
- If first player checks, second player can check or bet
- If first player bets, second player can call (costs 1 chip) or fold
- If second player bets after first player's check, first player can call or fold
- At showdown, the player with the higher card wins the pot

The key difference is that instead of using just 3 cards (Jack, Queen, King), we use a 100-card deck where the values Jack (0), Queen (1), and King (2) repeat throughout the deck. This creates a much larger strategy space due to positional effects.

## Card Abstraction

To manage the large state space, the implementation uses a bucketing system to abstract the 100 cards into a smaller number of strategic equivalence classes. The default configuration uses 10 buckets, but this can be adjusted in the config.py file.

In this implementation:
- The physical deck has 100 cards
- Each card has a logical value of Jack (0), Queen (1), or King (2)
- When using more than 3 buckets, we're creating position-based abstraction where the same card value in different deck positions can be treated differently

## CFR Implementation

The project includes multiple CFR variants:
- Vanilla CFR: The standard implementation of Counterfactual Regret Minimization
- CFR+: An enhanced version that uses regret matching+ and discounting
- Linear CFR: A variant that applies linear weighting to regret updates

## Running the Game

To train new strategies:
```
python Main.py
```

To play against the AI:
```
python kuhn_poker_game.py
```

To analyze existing strategies:
```
python debug_strategy.py [strategy_file]
```

## Configuration

Adjust the parameters in config.py to customize:
- Card abstraction (buckets)
- Training iterations
- CFR variant
- Early stopping criteria
- And more

## Strategy Analysis

The project includes tools to visualize and analyze trained strategies. After training, strategy visualization will be saved to the 'visualization' directory, showing:
- Exploitability comparison
- Head-to-head performance
- Action probabilities by card bucket

## Requirements

See requirements.txt for dependencies. The main requirements are:
- numpy
- tqdm
- matplotlib
- seaborn
- pandas
- pygame (for the interactive game)
