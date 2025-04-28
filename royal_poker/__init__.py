"""
Royal Poker Package

A Python implementation of Royal No Limit Hold'em poker environment and 
CFR-based AI agents with action abstractions.

Key Components:
- KuhnPoker: Simplified poker environment
- KuhnPokerEnv: Alias for KuhnPoker (for backward compatibility)
- EnhancedCFR: Counterfactual Regret Minimization solver
- LargeScaleCFR: CFR implementation for large state spaces
- ActionAbstraction: Action abstraction definitions
- KuhnEvaluator: Tools for evaluating strategies
"""

__version__ = "0.1.0"

# Import core components for easy access
from .env import KuhnPokerEnv
from .cfr import KuhnCFR
from .evaluate import Evaluator
from .utils import (
    hand_strength,
    action_translation,
    save_strategy,
    load_strategy
)

# Import agent implementations
from .agent import PokerAgent, CFRAgent, RuleBasedAgent

# Also import the incremental CFRAgent from the root module
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from royal_poker.cfrAgent import CFRAgent as IncrementalCFRAgent

__all__ = [
    'KuhnPokerEnv',
    'KuhnCFR',
    'Evaluator',
    'hand_strength',
    'action_translation',
    'save_strategy',
    'load_strategy',
    'PokerAgent',
    'CFRAgent',
    'RuleBasedAgent',
    'IncrementalCFRAgent'
]