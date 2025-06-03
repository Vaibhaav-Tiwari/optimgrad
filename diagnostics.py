import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from engine import Tensor
from layers import Module

class GradientTracker:
    def __init__(self, model: Module):
        self.model = model
        self.grad_history: Dict[str, List[float]] = defaultdict(list)
        self.activation_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        
    def track_gradients(self) -> Dict[str, float]:
        """Compute and store gradient norms for all parameters."""
        grad_norms = {}
        
        for i, param in enumerate(self.model.parameters()):
            if param.grad is not None:
                norm = float(np.sqrt(np.sum(param.grad ** 2)))
                name = f"param_{i}"
                grad_norms[name] = norm
                self.grad_history[name].append(norm)
                
        return grad_norms
    
    def track_activations(self, layer_outputs: Dict[str, Tensor]) -> None:
        """Store activation statistics for each layer."""
        for name, output in layer_outputs.items():
            stats = {
                'mean': float(np.mean(output.data)),
                'std': float(np.std(output.data)),
                'min': float(np.min(output.data)),
                'max': float(np.max(output.data))
            }
            self.activation_history[name].append(stats)

def detect_gradient_pathologies(grad_history: Dict[str, List[float]], 
                              threshold_low: float = 1e-6,
                              threshold_high: float = 1e2) -> Dict[str, List[str]]:
    """Detect vanishing or exploding gradients.
    
    Args:
        grad_history: Dictionary of gradient norms over time
        threshold_low: Threshold for vanishing gradients
        threshold_high: Threshold for exploding gradients
    
    Returns:
        Dictionary of warnings per parameter
    """
    warnings = defaultdict(list)
    
    for param_name, norms in grad_history.items():
        if not norms:
            continue
            
        recent_norms = norms[-10:]  # Look at last 10 iterations
        mean_norm = np.mean(recent_norms)
        
        if mean_norm < threshold_low:
            warnings[param_name].append(
                f"Vanishing gradient detected: mean norm = {mean_norm:.2e}"
            )
        elif mean_norm > threshold_high:
            warnings[param_name].append(
                f"Exploding gradient detected: mean norm = {mean_norm:.2e}"
            )
            
    return dict(warnings)

class ActivationHook:
    """Hook to track layer activations during forward pass."""
    
    def __init__(self, layer: Module, name: str):
        self.layer = layer
        self.name = name
        self.activations = []
        self.original_forward = layer.forward
        
        # Replace forward method with hooked version
        def forward_hook(x: Tensor) -> Tensor:
            output = self.original_forward(x)
            self.activations.append(output.data.copy())
            return output
            
        layer.forward = forward_hook
    
    def clear(self):
        """Clear stored activations."""
        self.activations = []
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute statistics of stored activations."""
        if not self.activations:
            return {}
            
        all_activations = np.concatenate([a.reshape(-1) for a in self.activations])
        return {
            'mean': float(np.mean(all_activations)),
            'std': float(np.std(all_activations)),
            'min': float(np.min(all_activations)),
            'max': float(np.max(all_activations)),
            'fraction_dead': float(np.mean(all_activations == 0))
        } 