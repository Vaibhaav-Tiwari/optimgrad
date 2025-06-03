# OptimGrad: Technical Documentation

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Core Components](#2-core-components)
3. [Scientific Utilities](#3-scientific-utilities)
4. [Diagnostic Tools](#4-diagnostic-tools)
5. [Neural Network Components](#5-neural-network-components)
6. [Training Infrastructure](#6-training-infrastructure)
7. [Testing Framework](#7-testing-framework)
8. [Project Dependencies](#8-project-dependencies)
9. [Mathematical Foundation](#9-mathematical-foundation)

## 1. Project Overview

OptimGrad is an educational automatic differentiation engine that implements reverse-mode automatic differentiation (backpropagation) for deep learning. The project is structured into several key components:

```
micrograd/
├── engine.py        # Core autograd engine
├── scientific.py    # Scientific computing utilities
├── diagnostics.py   # Diagnostic and monitoring tools
├── layers.py        # Neural network layers
├── models.py        # Model architectures
├── losses.py        # Loss functions
├── train.py        # Training utilities
├── viz.py          # Visualization tools
└── test_*.py       # Test files
```

## 2. Core Components (engine.py)

### Tensor Class
The fundamental building block of the library.

```python
class Tensor:
    def __init__(self, data, requires_grad=False):
        self._data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self._data)
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self.children = set()
        self._in_graph = False
```

Key Features:
- Automatic gradient computation
- Dynamic computational graph building
- Broadcasting support
- In-place operation detection

Mathematical Operations:
1. Addition (`__add__`):
   ```python
   def __add__(self, other):
       out = Tensor(self.data + other.data)
       if self.requires_grad or other.requires_grad:
           def _backward():
               if self.requires_grad:
                   self.grad += out.grad
               if other.requires_grad:
                   other.grad += out.grad
           out._backward = _backward
   ```

2. Multiplication (`__mul__`):
   ```python
   def __mul__(self, other):
       out = Tensor(self.data * other.data)
       if self.requires_grad or other.requires_grad:
           def _backward():
               if self.requires_grad:
                   self.grad += other.data * out.grad
               if other.requires_grad:
                   other.grad += self.data * out.grad
           out._backward = _backward
   ```

## 3. Scientific Utilities (scientific.py)

### Gradient Clipping
Prevents exploding gradients by scaling them when their norm exceeds a threshold.

```python
def clip_gradients(params: List[Tensor], max_norm: float) -> None:
    total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in params))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale
```

Mathematical formulation:
\[
\text{scale} = \frac{\text{max\_norm}}{\sqrt{\sum_{i} \|\nabla \theta_i\|^2}}
\]
\[
\nabla \theta_i \leftarrow \text{scale} \cdot \nabla \theta_i
\]

### Bound Estimation
Estimates output bounds of a function using interval arithmetic.

```python
def estimate_bounds(f: Callable[[Tensor], Tensor], x: Tensor, 
                   eps: float = 1e-4) -> Tuple[float, float]:
```

Method:
1. Perturbs each input dimension by ±eps
2. Evaluates function at perturbed points
3. Returns minimum and maximum observed values

## 4. Diagnostic Tools (diagnostics.py)

### GradientTracker
Monitors gradient norms during training.

```python
class GradientTracker:
    def track_gradients(self) -> Dict[str, float]:
        grad_norms = {}
        for i, param in enumerate(self.model.parameters()):
            norm = float(np.sqrt(np.sum(param.grad ** 2)))
            grad_norms[f"param_{i}"] = norm
```

### ActivationHook
Monitors layer activations during forward pass.

```python
class ActivationHook:
    def get_statistics(self) -> Dict[str, float]:
        return {
            'mean': float(np.mean(self.activations)),
            'std': float(np.std(self.activations)),
            'fraction_dead': float(np.mean(self.activations == 0))
        }
```

### Gradient Pathology Detection
Identifies vanishing and exploding gradients.

```python
def detect_gradient_pathologies(grad_history: Dict[str, List[float]],
                              threshold_low: float = 1e-6,
                              threshold_high: float = 1e2) -> Dict[str, List[str]]
```

## 5. Neural Network Components

### Layers (layers.py)
1. Linear Layer:
   ```python
   class Linear(Module):
       def __init__(self, in_features: int, out_features: int):
           self.weight = Tensor(xavier_init((in_features, out_features)))
           self.bias = Tensor(np.zeros(out_features))
   ```

2. Activation Functions:
   - ReLU:
     ```math
     f(x) = \max(0, x)
     ```
     ```math
     f'(x) = \begin{cases} 
     1 & \text{if } x > 0 \\
     0 & \text{otherwise}
     \end{cases}
     ```

   - Sigmoid:
     ```math
     f(x) = \frac{1}{1 + e^{-x}}
     ```
     ```math
     f'(x) = f(x)(1 - f(x))
     ```

   - Tanh:
     ```math
     f(x) = \tanh(x)
     ```
     ```math
     f'(x) = 1 - \tanh^2(x)
     ```

3. BatchNorm1D:
   ```python
   class BatchNorm1D(Module):
       def __init__(self, num_features: int, eps: float = 1e-5):
           self.gamma = Tensor(np.ones(num_features))
           self.beta = Tensor(np.zeros(num_features))
   ```

### Models (models.py)
1. Sequential:
   ```python
   class Sequential(Module):
       def forward(self, x: Tensor) -> Tensor:
           for layer in self.layers:
               x = layer(x)
           return x
   ```

2. MLP (Multi-Layer Perceptron):
   ```python
   class MLP(Module):
       def __init__(self, layer_sizes: List[int], activation: str = 'relu'):
           # Builds layers with specified sizes and activation
   ```

## 6. Training Infrastructure (train.py)

### Optimizers
1. SGD:
   ```python
   class SGD(Optimizer):
       def step(self):
           for p in self.params:
               p.data -= self.lr * p.grad
   ```

2. Adam:
   ```python
   class Adam(Optimizer):
       def step(self):
           self.t += 1
           for i, p in enumerate(self.params):
               # Update biased first moment estimate
               self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
               # Update biased second raw moment estimate
               self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
   ```

### DataLoader
```python
class DataLoader:
    def __iter__(self) -> Iterator[Tuple[Tensor, Optional[Tensor]]]:
        for start_idx in range(0, len(self.X), self.batch_size):
            yield (Tensor(self.X[start_idx:start_idx + self.batch_size]),
                  Tensor(self.y[start_idx:start_idx + self.batch_size]))
```

## 7. Testing Framework

### test_scientific.py
Tests for scientific computing utilities:
- Gradient clipping
- Bound estimation
- Gradient tracking
- Activation monitoring
- Pathology detection

Example test:
```python
def test_gradient_clipping():
    params = [Tensor([[1.0, 2.0]], requires_grad=True)]
    params[0].grad = np.array([[10.0, 20.0]])
    clip_gradients(params, max_norm=1.0)
    total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in params))
    assert np.abs(total_norm - 1.0) < 1e-6
```

## 8. Project Dependencies

Required packages:
```
numpy>=1.19.0
matplotlib>=3.3.0
tqdm>=4.50.0
```

## 9. Mathematical Foundation

### Automatic Differentiation
The library implements reverse-mode automatic differentiation, which efficiently computes gradients by:
1. Forward pass: Building computational graph
2. Backward pass: Applying chain rule in reverse order

Chain Rule Implementation:
\[
\frac{\partial L}{\partial x} = \sum_i \frac{\partial L}{\partial y_i} \frac{\partial y_i}{\partial x}
\]

### Loss Functions
1. Mean Squared Error (MSE):
   ```math
   L = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2
   ```

2. Binary Cross-Entropy (BCE):
   ```math
   L = -\frac{1}{n}\sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]
   ```

### Optimization
1. SGD Update Rule:
   ```math
   \theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
   ```

2. Adam Update Rules:
   ```math
   m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla_\theta L(\theta_t)
   ```
   ```math
   v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla_\theta L(\theta_t))^2
   ```
   ```math
   \hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
   ```
   ```math
   \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon}\hat{m}_t
   ```

## Usage Examples

### Basic Tensor Operations
```python
# Create tensors
x = Tensor([2.0], requires_grad=True)
y = Tensor([3.0], requires_grad=True)

# Forward pass
z = x * y + x**2

# Backward pass
z.backward()

# Access gradients
print(x.grad)  # dy/dx = y + 2x = 3 + 4 = 7
print(y.grad)  # dy/dy = x = 2
```

### Training a Neural Network
```python
# Create model
model = Sequential([
    Linear(2, 4),
    ReLU(),
    Linear(4, 1)
])

# Create optimizer
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(num_epochs):
    for X_batch, y_batch in train_loader:
        # Forward pass
        y_pred = model(X_batch)
        loss = mse_loss(y_pred, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Using Diagnostic Tools
```python
# Track gradients
tracker = GradientTracker(model)
grad_norms = tracker.track_gradients()

# Monitor activations
hook = ActivationHook(model.layers[0], "linear1")
stats = hook.get_statistics()

# Check for gradient pathologies
warnings = detect_gradient_pathologies(tracker.grad_history)
``` 