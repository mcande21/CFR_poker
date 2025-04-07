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

__all__ = [
    'KuhnPokerEnv',
    'KuhnCFR',
    'Evaluator',
    'hand_strength',
    'action_translation',
    'save_strategy',
    'load_strategy'
]