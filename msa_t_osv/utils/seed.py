"""Random seed utilities for reproducibility."""

import random
import os
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, cuda_deterministic: bool = True):
    """Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
        cuda_deterministic: Whether to use deterministic CUDA operations
    """
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    
    # Environment variable for some libraries
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # CUDA deterministic operations
    if cuda_deterministic:
        set_deterministic(True)
        

def set_deterministic(deterministic: bool = True):
    """Set PyTorch to use deterministic algorithms.
    
    Note: This may impact performance.
    
    Args:
        deterministic: Whether to use deterministic algorithms
    """
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    
    # For PyTorch >= 1.8
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(deterministic)
        except:
            # Some operations might not have deterministic implementations
            pass
            
    # Set CUBLAS workspace config for determinism
    if deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        

def get_random_state() -> dict:
    """Get current random state for all random number generators.
    
    Returns:
        Dictionary containing random states
    """
    state = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_random': torch.get_rng_state(),
    }
    
    if torch.cuda.is_available():
        state['torch_cuda_random'] = torch.cuda.get_rng_state_all()
        
    return state
    

def set_random_state(state: dict):
    """Restore random state for all random number generators.
    
    Args:
        state: Dictionary containing random states
    """
    if 'python_random' in state:
        random.setstate(state['python_random'])
        
    if 'numpy_random' in state:
        np.random.set_state(state['numpy_random'])
        
    if 'torch_random' in state:
        torch.set_rng_state(state['torch_random'])
        
    if 'torch_cuda_random' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda_random'])
        

class RandomContext:
    """Context manager for temporary random seed.
    
    Usage:
        with RandomContext(seed=42):
            # Code with fixed random seed
            pass
        # Random state restored
    """
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize random context.
        
        Args:
            seed: Random seed to use. If None, current state is maintained.
        """
        self.seed = seed
        self.state = None
        
    def __enter__(self):
        """Enter context and save current state."""
        self.state = get_random_state()
        if self.seed is not None:
            set_seed(self.seed)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and restore previous state."""
        if self.state is not None:
            set_random_state(self.state) 