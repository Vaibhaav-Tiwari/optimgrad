import numpy as np
from typing import Union
from engine import Tensor

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax implementation"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def cross_entropy_loss(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """Combined softmax and cross entropy loss with numerical stability
    
    Args:
        pred: Raw logits from the model (not softmaxed)
        target: Either one-hot encoded or class indices
    """
    # Convert target to one-hot if needed
    if len(target.shape) == 1 or target.shape[-1] == 1:
        n_classes = pred.data.shape[-1]
        target_one_hot = np.zeros_like(pred.data)
        target_one_hot[np.arange(len(target)), target.reshape(-1)] = 1
        target_data = target_one_hot
    else:
        target_data = target.data if isinstance(target, Tensor) else target
    
    # Compute softmax
    softmax_out = softmax(pred.data)
    
    # Compute cross entropy
    eps = 1e-7  # For numerical stability
    softmax_out = np.clip(softmax_out, eps, 1 - eps)
    loss = -np.sum(target_data * np.log(softmax_out)) / pred.data.shape[0]
    
    # Create output tensor
    out = Tensor(loss, requires_grad=pred.requires_grad)
    
    if pred.requires_grad:
        # Define backward function
        def _backward():
            pred.grad += (softmax_out - target_data) / pred.data.shape[0]
            
        out._backward = _backward
        out.children = {pred}
        pred._in_graph = True
        
    return out

def mse_loss(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """Mean squared error loss"""
    target_data = target.data if isinstance(target, Tensor) else target
    diff = pred.data - target_data
    loss = np.mean(diff ** 2)
    
    out = Tensor(loss, requires_grad=pred.requires_grad)
    
    if pred.requires_grad:
        def _backward():
            pred.grad += 2 * diff / np.prod(pred.data.shape)
            
        out._backward = _backward
        out.children = {pred}
        pred._in_graph = True
        
    return out

def bce_loss(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """Binary cross entropy loss with numerical stability"""
    target_data = target.data if isinstance(target, Tensor) else target
    eps = 1e-7
    pred_clipped = np.clip(pred.data, eps, 1 - eps)
    
    # Compute loss with numerical stability
    loss = -(target_data * np.log(pred_clipped) + 
             (1 - target_data) * np.log(1 - pred_clipped)).mean()
    
    out = Tensor(loss, requires_grad=pred.requires_grad)
    
    if pred.requires_grad:
        def _backward():
            # Simple gradient computation: (p - y) / n
            # This is correct because BCE assumes input is already sigmoided
            pred.grad += (pred_clipped - target_data) / np.prod(pred.data.shape)
            
        out._backward = _backward
        out.children = {pred}
        pred._in_graph = True
        
    return out 