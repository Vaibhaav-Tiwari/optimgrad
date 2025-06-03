from typing import Protocol, List, Tuple, Optional, Callable, Any, Dict, Union, Iterator
import numpy as np
from engine import Tensor
from layers import Module
import time
from tqdm import tqdm

# Protocol for models
class Model(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...
    def parameters(self) -> List[Tensor]: ...

# Base optimizer class
class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params
    
    def zero_grad(self):
        for p in self.params:
            p.grad = np.zeros_like(p.data)
    
    def step(self):
        raise NotImplementedError

# SGD optimizer
class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01,
                 momentum: float = 0.0, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        for i, p in enumerate(self.params):
            if p.requires_grad:
                grad = p.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * p.data
                    
                if self.momentum != 0:
                    self.velocities[i] = (
                        self.momentum * self.velocities[i] + grad
                    )
                    grad = self.velocities[i]
                    
                # Update parameters (without in-place operation)
                p._data = p.data - self.lr * grad

# Adam optimizer
class Adam(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.001,
                 betas: Tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # Initialize momentum and velocity
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]
    
    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.requires_grad:
                grad = p.grad
                if self.weight_decay != 0:
                    grad = grad + self.weight_decay * p.data
                
                # Update biased first moment estimate
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * grad
                
                # Update biased second raw moment estimate
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (grad * grad)
                
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.betas[0]**self.t)
                
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.betas[1]**self.t)
                
                # Update parameters (without in-place operation)
                p._data = p.data - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

# DataLoader for mini-batching
class DataLoader:
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None,
                 batch_size: int = 32, shuffle: bool = True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def __iter__(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        indices = np.arange(len(self.X))
        if self.shuffle:
            np.random.shuffle(indices)
            
        for start_idx in range(0, len(indices), self.batch_size):
            batch_idx = indices[start_idx:start_idx + self.batch_size]
            X_batch = Tensor(self.X[batch_idx])
            if self.y is not None:
                y_batch = Tensor(self.y[batch_idx])
            else:
                y_batch = None
            yield X_batch, y_batch

class LRScheduler:
    def __init__(self, optimizer: 'Optimizer', mode: str = 'step',
                 gamma: float = 0.1, step_size: int = 10,
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.gamma = gamma
        self.step_size = step_size
        self.min_lr = min_lr
        self.base_lr = optimizer.lr
        self.last_epoch = 0
        
    def step(self, epoch: Optional[int] = None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if self.mode == 'step':
            if epoch % self.step_size == 0:
                self.optimizer.lr = max(
                    self.optimizer.lr * self.gamma,
                    self.min_lr
                )
        elif self.mode == 'exp':
            self.optimizer.lr = max(
                self.base_lr * (self.gamma ** epoch),
                self.min_lr
            )
        else:
            raise ValueError(f"Unknown scheduler mode: {self.mode}")

def train_epoch(model: Module,
                dataloader: DataLoader,
                loss_fn: Callable,
                optimizer: Optimizer) -> Dict[str, float]:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in dataloader:
        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data
        num_batches += 1
    
    return {'loss': total_loss / num_batches}

def evaluate(model: Module,
             dataloader: DataLoader,
             loss_fn: Callable) -> Dict[str, float]:
    """Evaluate model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for X_batch, y_batch in dataloader:
        # Forward pass without computing gradients
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        total_loss += loss.data
        num_batches += 1
    
    return {'loss': total_loss / num_batches}

def train(model: Module,
          train_loader: DataLoader,
          val_loader: Optional[DataLoader],
          loss_fn: Callable,
          optimizer: Optimizer,
          scheduler: Optional[LRScheduler] = None,
          num_epochs: int = 10,
          verbose: bool = True) -> Dict[str, List[float]]:
    """Full training loop with validation"""
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        train_metrics = train_epoch(model, train_loader, loss_fn, optimizer)
        history['train_loss'].append(train_metrics['loss'])
        
        # Validation phase
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, loss_fn)
            history['val_loss'].append(val_metrics['loss'])
            
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if val_loader is not None:
                print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Learning Rate: {optimizer.lr:.6f}")
    
    return history 