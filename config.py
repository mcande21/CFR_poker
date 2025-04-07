class Config:
    # CFR parameters
    CFR_ITERATIONS = 250_000  # Increased for better convergence
    CFR_VARIANT = "CFR+"     # Keep CFR+ as it's already optimal
    
    # Optimization flags
    USE_PRUNING = True
    PRUNING_THRESHOLD = 5_000  # More frequent pruning
    WEIGHTED_UPDATES = True
    
    # Card abstraction
    BUCKETS = 3      # 3 buckets representing Jack(0), Queen(1), King(2)
    DECK_SIZE = 3    # 3 card types in Kuhn poker
    CARD_VALUES = 3  # Jack=0, Queen=1, King=2
    
    # Early stopping
    EARLY_STOPPING = True
    CONVERGENCE_THRESHOLD = 1e-8  # Tighter convergence for more precise strategy
    PATIENCE = 5000  # Increased patience to ensure proper convergence
    
    # Evaluation settings
    EVAL_FREQUENCY = 10000  # Less frequent evaluation to speed up training
    
    # Additional recommended parameters
    USE_LINEAR_AVERAGING = True  # Better averaging scheme for CFR+
    ALTERNATING_UPDATES = True   # Alternating player updates for faster convergence
