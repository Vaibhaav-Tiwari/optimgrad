import numpy as np
from typing import Callable
from engine import Tensor

# Loss functions with numerically stable implementations
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    return ((pred - target) ** 2).mean()

def bce_loss(pred: Tensor, target: Tensor) -> Tensor:
    eps = 1e-7
    p = pred.data.clip(eps, 1 - eps)
    out = Tensor(-(target.data * np.log(p) + (1 - target.data) * np.log(1 - p)).mean(),
                requires_grad=pred.requires_grad)
    
    def _backward():
        if pred.requires_grad:
            pred.grad += (pred.data - target.data) / (pred.data * (1 - pred.data)) * out.grad / pred.data.size
            
    out._backward_fn = _backward
    out.children = (pred, target)
    out.op = 'BCE'
    return out

# Gradient checking utilities
def grad_check(f: Callable[[Tensor], Tensor], x: Tensor, eps: float = 1e-7) -> bool:
    analytical = np.zeros_like(x.data)
    numerical = np.zeros_like(x.data)
    
    # Compute analytical gradient
    out = f(x)
    out.backward()
    analytical = x.grad.copy()
    x.grad = np.zeros_like(x.data)  # Reset grad
    
    # Compute numerical gradient
    it = np.nditer(x.data, flags=['multi_index'])
    while not it.finished:
        ix = it.multi_index
        old_val = x.data[ix]
        
        x.data[ix] = old_val + eps
        out_plus = f(x).data
        
        x.data[ix] = old_val - eps
        out_minus = f(x).data
        
        x.data[ix] = old_val
        numerical[ix] = (out_plus - out_minus) / (2 * eps)
        
        it.iternext()
    
    # Compare gradients
    max_diff = np.max(np.abs(analytical - numerical))
    return max_diff < 1e-5

def check_grad_flow(tensor: Tensor, threshold_min: float = 1e-6, threshold_max: float = 1e3) -> None:
    if np.any(np.abs(tensor.grad) > threshold_max):
        raise Warning(f"Gradient explosion detected! Max gradient: {np.max(np.abs(tensor.grad))}")
    if np.any(np.abs(tensor.grad) < threshold_min) and np.any(tensor.grad != 0):
        raise Warning(f"Gradient vanishing detected! Min non-zero gradient: {np.min(np.abs(tensor.grad[tensor.grad != 0]))}") 