import numpy as np
from typing import Callable, List, Tuple, Optional
from engine import Tensor

def clip_gradients(params: List[Tensor], max_norm: float) -> None:
    """Clip gradients to have a maximum norm.
    
    Args:
        params: List of parameters whose gradients should be clipped
        max_norm: Maximum allowed norm of the gradients
    """
    # Compute total norm of all gradients
    total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in params if p.grad is not None))
    
    if total_norm > max_norm:
        # Scale gradients to have norm = max_norm
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad *= scale

def estimate_bounds(f: Callable[[Tensor], Tensor], x: Tensor, 
                   eps: float = 1e-4) -> Tuple[float, float]:
    """Estimate output bounds of function using interval arithmetic.
    
    Args:
        f: Function to analyze
        x: Input tensor
        eps: Small perturbation for sampling
        
    Returns:
        (lower_bound, upper_bound) estimates
    """
    # Save original data
    orig_data = x.data.copy()
    results = []
    
    # Sample points around x
    x_flat = x.data.flatten()
    for i in range(len(x_flat)):
        # Create perturbed versions
        x_plus = x.data.copy()
        x_minus = x.data.copy()
        
        # Perturb one dimension at a time
        x_plus.flat[i] += eps
        x_minus.flat[i] -= eps
        
        # Evaluate function
        x.data = x_plus
        results.append(f(x).data)
        x.data = x_minus
        results.append(f(x).data)
    
    # Restore original data
    x.data = orig_data
    
    # Compute bounds
    results = np.concatenate([r.flatten() for r in results])
    return float(np.min(results)), float(np.max(results)) 