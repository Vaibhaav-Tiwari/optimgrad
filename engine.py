import math
import numpy as np
from typing import List, Tuple, Callable, Union, Optional
from functools import partial

# Core autograd engine implementing reverse mode automatic differentiation
class Tensor:
    def __init__(self, data, requires_grad=False):
        if isinstance(data, Tensor):
            self._data = data.data
        else:
            self._data = np.array(data, dtype=np.float32)
        
        self.grad = np.zeros_like(self._data)  # Always initialize grad to zeros
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self.children = set()
        self._in_graph = False
        self.op = ''

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self._in_graph:
            raise RuntimeError("Cannot modify tensor data that is part of a computation graph")
        self._data = np.array(value, dtype=np.float32)

    @property
    def T(self):
        """Return the transpose of this tensor."""
        out = Tensor(self.data.T, requires_grad=self.requires_grad)
        
        def _backward():
            if out.grad is not None:
                if self.grad is None:
                    self.grad = np.zeros_like(self.data)
                self.grad += out.grad.T
        
        if self.requires_grad:
            out._backward = _backward
            out.children = {self}
            self._in_graph = True
        
        return out

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
    
    def zero_grad(self):
        """Reset gradients to zero"""
        self.grad = np.zeros_like(self._data)

    def backward(self, grad=None):
        if not self.requires_grad:
            return
            
        if grad is None:
            if self.data.size == 1:  # scalar output (size of 1)
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("grad must be specified for non-scalar outputs")
        
        self.grad = grad
        
        # Build topological order of all nodes in the graph
        topo = []
        visited = set()
        
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    build_topo(child)
                topo.append(node)
        
        build_topo(self)
        
        # Go one variable at a time and apply the chain rule to get its gradient
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting in backward pass
                grad = out.grad
                if self.data.shape != grad.shape:
                    # Sum gradients along broadcasted dimensions
                    for _ in range(len(grad.shape) - len(self.data.shape)):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(self.data.shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
            if other.requires_grad:
                # Handle broadcasting in backward pass
                grad = out.grad
                if other.data.shape != grad.shape:
                    # Sum gradients along broadcasted dimensions
                    for _ in range(len(grad.shape) - len(other.data.shape)):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(other.data.shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
                
        out._backward = _backward
        out.children = {self, other}
        
        # Mark tensors as part of computation graph
        if out.requires_grad:
            self._in_graph = True
            other._in_graph = True
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                # Handle broadcasting in backward pass
                grad = other.data * out.grad
                if self.data.shape != grad.shape:
                    # Sum gradients along broadcasted dimensions
                    for _ in range(len(grad.shape) - len(self.data.shape)):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(self.data.shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                self.grad += grad
            if other.requires_grad:
                # Handle broadcasting in backward pass
                grad = self.data * out.grad
                if other.data.shape != grad.shape:
                    # Sum gradients along broadcasted dimensions
                    for _ in range(len(grad.shape) - len(other.data.shape)):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(other.data.shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                other.grad += grad
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {self, other}
            self._in_graph = True
            other._in_graph = True
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other ** -1)

    def __pow__(self, other: float):
        assert isinstance(other, (int, float)), "Only supporting int/float powers for now"
        out = Tensor(self.data ** other, requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += other * (self.data ** (other - 1)) * out.grad
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {self}
            self._in_graph = True
        return out

    def sum(self):
        out = Tensor(np.sum(self.data), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad
                
        out._backward = _backward
        out.children = {self}
        out.op = 'sum'
        
        # Mark tensor as part of computation graph
        self._in_graph = True
        return out

    def mean(self):
        out = Tensor(np.mean(self.data), requires_grad=self.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += np.ones_like(self.data) * out.grad / self.data.size
                
        out._backward = _backward
        out.children = {self}
        out.op = 'mean'
        
        # Mark tensor as part of computation graph
        self._in_graph = True
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.matmul(self.data, other.data), requires_grad=self.requires_grad or other.requires_grad)
        
        def _backward():
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.matmul(self.data.T, out.grad)
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {self, other}
            self._in_graph = True
            other._in_graph = True
        return out

    def __rmatmul__(self, other) -> 'Tensor':
        """Reverse matrix multiplication"""
        return Tensor(other) @ self

# Activation functions and utilities
def relu(x: Tensor) -> Tensor:
    out = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            x.grad += (out.data > 0) * out.grad
            
    out._backward = _backward
    out.children = {x}
    out.op = 'ReLU'
    
    # Mark tensor as part of computation graph
    x._in_graph = True
    return out

def sigmoid(x: Tensor) -> Tensor:
    out = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            x.grad += out.data * (1 - out.data) * out.grad
            
    out._backward = _backward
    out.children = {x}
    out.op = 'sigmoid'
    
    # Mark tensor as part of computation graph
    x._in_graph = True
    return out

def tanh(x: Tensor) -> Tensor:
    out = Tensor(np.tanh(x.data), requires_grad=x.requires_grad)
    
    def _backward():
        if x.requires_grad:
            x.grad += (1 - out.data**2) * out.grad
            
    out._backward = _backward
    out.children = {x}
    out.op = 'tanh'
    
    # Mark tensor as part of computation graph
    x._in_graph = True
    return out

# Dropout layer
class Dropout:
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None
        self.training = True
        
    def __call__(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0:
            return x
            
        self.mask = np.random.binomial(1, 1-self.p, x.data.shape) / (1-self.p)
        out = Tensor(x.data * self.mask, requires_grad=x.requires_grad)
        
        def _backward():
            if x.requires_grad:
                x.grad += self.mask * out.grad
                
        if out.requires_grad:
            out._backward = _backward
            out.children = {x}
            out.op = 'dropout'
            x._in_graph = True
            
        return out
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False 